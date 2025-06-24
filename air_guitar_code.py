import cv2
import mediapipe as mp
import pygame

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Load chord sounds
chords = {
    "G": pygame.mixer.Sound("chords/chord-g-major.wav"),
    "C": pygame.mixer.Sound("chords/chord-cadd9.wav"),
    "E": pygame.mixer.Sound("chords/chord-e-minor.wav"),
    "D": pygame.mixer.Sound("chords/chord-d-major.wav"),
}

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)
tip_ids = [4, 8, 12, 16, 20]
last_chord = None


def get_finger_pattern(landmarks):
    return [
        landmarks.landmark[tip_ids[i]].y < landmarks.landmark[tip_ids[i] - 2].y
        for i in range(1, 5)  # Index to pinky (skip thumb)
    ]


def detect_chord(fingers):
    if fingers == [True, False, False, False]:
        return "G"
    if fingers == [True, True, False, False]:
        return "C"
    if fingers == [True, True, True, False]:
        return "E"
    if fingers == [True, True, True, True]:
        return "D"
    return None


while True:
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    chord = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        pattern = get_finger_pattern(hand)
        chord = detect_chord(pattern)

        cv2.putText(
            frame,
            f"Pattern: {pattern}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    # Play/stop chord based on detection
    if chord and chord != last_chord:
        if last_chord:
            chords[last_chord].stop()
        chords[chord].play(-1)
        last_chord = chord
        print(f"🎵 Playing: {chord}")

    if chord is None and last_chord:
        chords[last_chord].stop()
        last_chord = None

    cv2.putText(
        frame,
        f"Chord: {last_chord or 'None'}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Air Guitar", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
