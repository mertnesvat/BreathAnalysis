#!/usr/bin/env python3
"""
Breath Analysis Data Receiver

Connects to ESP32 via WebSocket, records sensor data during meditation sessions,
and saves to CSV for later analysis.

Usage:
    python receiver.py --ip 192.168.1.xxx
    python receiver.py --ip 192.168.1.xxx --duration 600  # 10 minute session
"""

import asyncio
import argparse
import json
import csv
import signal
import sys
from datetime import datetime
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


class BreathDataReceiver:
    def __init__(self, ip: str, port: int = 81):
        self.uri = f"ws://{ip}:{port}"
        self.data = []
        self.session_info = {}
        self.is_recording = False
        self.running = True

    async def connect_and_record(self, duration_seconds: int = None):
        """Connect to ESP32 and record data"""
        print(f"\n{'='*50}")
        print("  Breath Analysis - Data Receiver")
        print(f"{'='*50}\n")

        try:
            async with websockets.connect(self.uri) as ws:
                print(f"[Connected] {self.uri}")

                # Receive device info
                info_msg = await ws.recv()
                info = json.loads(info_msg)
                if info.get("type") == "info":
                    print(f"[Device] {info.get('device')}")
                    print(f"[Sample Rate] {info.get('sample_rate')} Hz")
                    print(f"[Sensors] {info.get('sensors')}")
                    self.session_info = info

                print(f"\n{'─'*50}")

                # Auto-start if duration specified and not interactive
                if duration_seconds and not sys.stdin.isatty():
                    print(f"[Auto-start] Duration specified, starting immediately...")
                else:
                    print("Press ENTER to start recording...")
                    print("Press Ctrl+C to stop and save")
                    print(f"{'─'*50}\n")
                    # Wait for user to press Enter
                    await asyncio.get_event_loop().run_in_executor(None, input)

                # Start recording
                await ws.send("start")
                self.is_recording = True
                start_time = datetime.now()
                print(f"[Recording] Started at {start_time.strftime('%H:%M:%S')}")

                if duration_seconds:
                    print(f"[Duration] Auto-stop after {duration_seconds} seconds")

                sample_count = 0
                try:
                    while self.running:
                        # Check duration limit
                        if duration_seconds:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            if elapsed >= duration_seconds:
                                print(f"\n[Duration] Reached {duration_seconds}s limit")
                                break

                        # Receive with timeout to allow checking running flag
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(msg)

                            if data.get("type") == "session_start":
                                print("[Session] Confirmed by device")
                            elif data.get("type") == "session_end":
                                print("[Session] Ended by device")
                                break
                            elif "t" in data:  # Sensor data
                                self.data.append(data)
                                sample_count += 1

                                # Progress indicator
                                if sample_count % 50 == 0:
                                    elapsed = data["t"] / 1000
                                    print(f"  Samples: {sample_count} | Time: {elapsed:.1f}s", end="\r")

                        except asyncio.TimeoutError:
                            continue

                except asyncio.CancelledError:
                    pass

                # Stop recording
                await ws.send("stop")
                self.is_recording = False
                end_time = datetime.now()

                print(f"\n\n[Recording] Stopped at {end_time.strftime('%H:%M:%S')}")
                print(f"[Samples] Total: {len(self.data)}")

        except ConnectionRefusedError:
            print(f"[Error] Cannot connect to {self.uri}")
            print("        Make sure ESP32 is powered and on the same network")
            return None
        except Exception as e:
            print(f"[Error] {e}")
            return None

        return self.save_session()

    def save_session(self) -> Path:
        """Save recorded data to CSV"""
        if not self.data:
            print("[Warning] No data to save")
            return None

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path(__file__).parent.parent / "data" / "sessions"
        data_dir.mkdir(parents=True, exist_ok=True)

        filename = data_dir / f"session_{timestamp}.csv"

        # Write CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Header with metadata as comment
            f.write(f"# Breath Analysis Session\n")
            f.write(f"# Date: {datetime.now().isoformat()}\n")
            f.write(f"# Sample Rate: {self.session_info.get('sample_rate', 'unknown')} Hz\n")
            f.write(f"# Total Samples: {len(self.data)}\n")
            f.write(f"# Duration: {self.data[-1]['t'] / 1000:.1f} seconds\n")
            f.write("#\n")

            # Column headers
            writer.writerow(["timestamp_ms", "thermistor", "piezo", "ir", "red"])

            # Data rows
            for row in self.data:
                writer.writerow([
                    row.get("t", 0),
                    row.get("th", 0),
                    row.get("pz", 0),
                    row.get("ir", 0),
                    row.get("rd", 0)
                ])

        print(f"\n[Saved] {filename}")
        print(f"        Size: {filename.stat().st_size / 1024:.1f} KB")

        return filename


async def main():
    parser = argparse.ArgumentParser(description="Record breath data from ESP32")
    parser.add_argument("--ip", required=True, help="ESP32 IP address")
    parser.add_argument("--port", type=int, default=81, help="WebSocket port (default: 81)")
    parser.add_argument("--duration", type=int, help="Auto-stop after N seconds")
    args = parser.parse_args()

    receiver = BreathDataReceiver(args.ip, args.port)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n[Interrupt] Stopping...")
        receiver.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Run receiver
    saved_file = await receiver.connect_and_record(args.duration)

    if saved_file:
        print(f"\n{'='*50}")
        print("  Session Complete!")
        print(f"{'='*50}")
        print(f"\nNext steps:")
        print(f"  1. Review data: python visualize.py {saved_file}")
        print(f"  2. Rate your session quality (1-10) and add notes")
        print()


if __name__ == "__main__":
    asyncio.run(main())
