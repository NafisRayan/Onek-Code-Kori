import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe face and hand modules
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize face detection
face_detection = mp_face_detection.FaceDetection()

# Initialize hand landmark detection
hands = mp_hands.Hands(max_num_hands=20)

# Load the glasses image with a transparent background
glasses = cv2.imread('f2.png', -1)

while True:
    # Read frames from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB and process it with Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    hand_results = hands.process(rgb)

    # Draw face landmarks and add glasses to the eyes
    if results.detections:
        for detection in results.detections:
            # Extract eye landmarks
            left_eye_landmark = detection.location_data.relative_keypoints[0]
            right_eye_landmark = detection.location_data.relative_keypoints[1]

            # Calculate the position and size of the glasses
            glasses_width = int(np.linalg.norm(np.array(left_eye_landmark) - np.array(right_eye_landmark)) * frame.shape[1])
            glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])
            glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

            # Calculate the position to overlay the glasses on the eyes
            x_offset = int(left_eye_landmark[0] * frame.shape[1])
            y_offset = int(left_eye_landmark[1] * frame.shape[0])

            # Overlay the glasses on the frame
            for c in range(0, 3):
                frame[y_offset:y_offset + glasses_resized.shape[0], x_offset:x_offset + glasses_resized.shape[1], c] = (
                    frame[y_offset:y_offset + glasses_resized.shape[0], x_offset:x_offset + glasses_resized.shape[1], c] * (1 - glasses_resized[:, :, 3] / 255.0) +
                    glasses_resized[:, :, c] * (glasses_resized[:, :, 3] / 255.0)
                )

            mp_drawing.draw_detection(frame, detection)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw circles around fingertips
            for landmark in hand_landmarks.landmark:
                if landmark.HasField('visibility') and landmark.visibility > 0.5:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow('Landmark Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
