#import packages
#pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np

# initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# take video input for pose detection
# you can put here video of your choice
cap = cv2.VideoCapture("2.mp4")

# take live camera  input for pose detection
#cap = cv2.VideoCapture(0)

# read each frame/image from capture object
while True:
    ret, img = cap.read()
    # Check if frame is empty
    if not ret:
        break

    # Resize image/frame so we can accommodate it on our screen
    img = cv2.resize(img, (600, 400))

    # Do Pose detection
    results = pose.process(img)

    # Check if any pose landmarks were detected
    if results.pose_landmarks:
        # Draw the detected pose for each person in the frame
        for person_landmarks in results.pose_landmarks:
            mp_draw.draw_landmarks(img, person_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                                   mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                                   )

    # Display pose on original video/live stream
    cv2.imshow("Pose Estimation", img)

    # Extract and draw pose on plain white image
    h, w, c = img.shape   # Get shape of original frame
    opImg = np.zeros([h, w, c])  # Create blank image with original frame size
    opImg.fill(255)  # Set white background. Put 0 if you want to make it black

    # Draw extracted pose on black white image
    if results.pose_landmarks:
        for person_landmarks in results.pose_landmarks:
            mp_draw.draw_landmarks(opImg, person_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                                   mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                                   )

    # Display extracted pose on blank images
    cv2.imshow("Extracted Pose", opImg)

    # Print the number of detected people
    if results.pose_landmarks != None:
        print("Number of people detected:", len(results.pose_landmarks))

    if cv2.waitKey(30) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

