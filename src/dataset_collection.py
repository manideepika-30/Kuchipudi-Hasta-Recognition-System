import cv2
import mediapipe as mp
import csv
import os

# Ask mudra name
label = input("Enter Mudra Name: ")

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# CSV file path
file_path = "data/mudra_dataset.csv"

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Drawing utility
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Open CSV file
with open(file_path, mode='a', newline='') as file:

    writer = csv.writer(file)

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera error")
            break

        # Mirror Effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand detection
        results = hands.process(rgb_frame)

        # If hand detected
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmark_list = []

                # Extract coordinates
                for landmark in hand_landmarks.landmark:

                    landmark_list.append(landmark.x)
                    landmark_list.append(landmark.y)

                # Add label
                landmark_list.append(label)

                # Save row in CSV
                writer.writerow(landmark_list)

                print(f"{label} saved")

        # Show frame
        cv2.imshow("Dataset Collection", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()