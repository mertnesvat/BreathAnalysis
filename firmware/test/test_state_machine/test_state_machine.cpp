/*
 * Unit Tests for Recording State Machine
 * Tests state transitions for recording/pause/stop
 * Runs on both device (ESP32) and native (computer)
 */

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <stdint.h>
#endif
#include <unity.h>

// ============== STATE MACHINE SIMULATION ==============
// Mirrors the state logic from main.cpp

class RecordingStateMachine {
public:
    bool isRecording = false;
    bool isPaused = false;
    bool sensorsOK = true;
    unsigned long sessionStartTime = 0;

    bool startRecording(unsigned long currentTime) {
        if (!sensorsOK) return false;

        isRecording = true;
        isPaused = false;
        sessionStartTime = currentTime;
        return true;
    }

    void stopRecording() {
        if (!isRecording) return;
        isRecording = false;
        isPaused = false;
    }

    void pause() {
        if (isRecording) {
            isPaused = true;
        }
    }

    void resume() {
        if (isRecording && isPaused) {
            isPaused = false;
        }
    }

    // Returns: 0=idle, 1=recording, 2=paused
    uint8_t getState() {
        if (!isRecording) return 0;
        return isPaused ? 2 : 1;
    }
};

RecordingStateMachine sm;

void setUp() {
    sm = RecordingStateMachine();
    sm.sensorsOK = true;
}

// ============== BASIC STATE TESTS ==============
void test_initial_state_idle() {
    TEST_ASSERT_FALSE(sm.isRecording);
    TEST_ASSERT_FALSE(sm.isPaused);
    TEST_ASSERT_EQUAL(0, sm.getState());
}

void test_start_recording() {
    bool result = sm.startRecording(1000);

    TEST_ASSERT_TRUE(result);
    TEST_ASSERT_TRUE(sm.isRecording);
    TEST_ASSERT_FALSE(sm.isPaused);
    TEST_ASSERT_EQUAL(1000, sm.sessionStartTime);
    TEST_ASSERT_EQUAL(1, sm.getState());
}

void test_start_recording_fails_without_sensors() {
    sm.sensorsOK = false;
    bool result = sm.startRecording(1000);

    TEST_ASSERT_FALSE(result);
    TEST_ASSERT_FALSE(sm.isRecording);
    TEST_ASSERT_EQUAL(0, sm.getState());
}

void test_stop_recording() {
    sm.startRecording(1000);
    sm.stopRecording();

    TEST_ASSERT_FALSE(sm.isRecording);
    TEST_ASSERT_FALSE(sm.isPaused);
    TEST_ASSERT_EQUAL(0, sm.getState());
}

void test_stop_when_not_recording() {
    // Should not crash or change state
    sm.stopRecording();
    TEST_ASSERT_FALSE(sm.isRecording);
    TEST_ASSERT_EQUAL(0, sm.getState());
}

// ============== PAUSE/RESUME TESTS ==============
void test_pause_while_recording() {
    sm.startRecording(1000);
    sm.pause();

    TEST_ASSERT_TRUE(sm.isRecording);
    TEST_ASSERT_TRUE(sm.isPaused);
    TEST_ASSERT_EQUAL(2, sm.getState());
}

void test_pause_when_not_recording() {
    sm.pause();

    TEST_ASSERT_FALSE(sm.isRecording);
    TEST_ASSERT_FALSE(sm.isPaused);
    TEST_ASSERT_EQUAL(0, sm.getState());
}

void test_resume_after_pause() {
    sm.startRecording(1000);
    sm.pause();
    sm.resume();

    TEST_ASSERT_TRUE(sm.isRecording);
    TEST_ASSERT_FALSE(sm.isPaused);
    TEST_ASSERT_EQUAL(1, sm.getState());
}

void test_resume_when_not_paused() {
    sm.startRecording(1000);
    sm.resume();  // Not paused, should be no-op

    TEST_ASSERT_TRUE(sm.isRecording);
    TEST_ASSERT_FALSE(sm.isPaused);
    TEST_ASSERT_EQUAL(1, sm.getState());
}

void test_resume_when_not_recording() {
    sm.resume();

    TEST_ASSERT_FALSE(sm.isRecording);
    TEST_ASSERT_FALSE(sm.isPaused);
}

// ============== STATE TRANSITION SEQUENCES ==============
void test_full_recording_cycle() {
    // Start -> Pause -> Resume -> Stop
    TEST_ASSERT_EQUAL(0, sm.getState());  // idle

    sm.startRecording(0);
    TEST_ASSERT_EQUAL(1, sm.getState());  // recording

    sm.pause();
    TEST_ASSERT_EQUAL(2, sm.getState());  // paused

    sm.resume();
    TEST_ASSERT_EQUAL(1, sm.getState());  // recording

    sm.stopRecording();
    TEST_ASSERT_EQUAL(0, sm.getState());  // idle
}

void test_stop_clears_paused_state() {
    sm.startRecording(1000);
    sm.pause();
    TEST_ASSERT_TRUE(sm.isPaused);

    sm.stopRecording();
    TEST_ASSERT_FALSE(sm.isPaused);
    TEST_ASSERT_FALSE(sm.isRecording);
}

void test_multiple_start_stop_cycles() {
    for (int i = 0; i < 3; i++) {
        sm.startRecording(i * 1000);
        TEST_ASSERT_TRUE(sm.isRecording);
        TEST_ASSERT_EQUAL(i * 1000, sm.sessionStartTime);

        sm.stopRecording();
        TEST_ASSERT_FALSE(sm.isRecording);
    }
}

void test_double_pause() {
    sm.startRecording(1000);
    sm.pause();
    sm.pause();  // Double pause should be safe

    TEST_ASSERT_TRUE(sm.isPaused);
    TEST_ASSERT_EQUAL(2, sm.getState());
}

void test_double_resume() {
    sm.startRecording(1000);
    sm.pause();
    sm.resume();
    sm.resume();  // Double resume should be safe

    TEST_ASSERT_FALSE(sm.isPaused);
    TEST_ASSERT_EQUAL(1, sm.getState());
}

// ============== TEST RUNNER ==============
void run_all_tests() {
    UNITY_BEGIN();

    // Basic state tests
    RUN_TEST(test_initial_state_idle);
    RUN_TEST(test_start_recording);
    RUN_TEST(test_start_recording_fails_without_sensors);
    RUN_TEST(test_stop_recording);
    RUN_TEST(test_stop_when_not_recording);

    // Pause/resume tests
    RUN_TEST(test_pause_while_recording);
    RUN_TEST(test_pause_when_not_recording);
    RUN_TEST(test_resume_after_pause);
    RUN_TEST(test_resume_when_not_paused);
    RUN_TEST(test_resume_when_not_recording);

    // Transition sequences
    RUN_TEST(test_full_recording_cycle);
    RUN_TEST(test_stop_clears_paused_state);
    RUN_TEST(test_multiple_start_stop_cycles);
    RUN_TEST(test_double_pause);
    RUN_TEST(test_double_resume);

    UNITY_END();
}

#ifdef ARDUINO
void setup() {
    delay(2000);
    run_all_tests();
}

void loop() {}
#else
int main() {
    run_all_tests();
    return 0;
}
#endif
