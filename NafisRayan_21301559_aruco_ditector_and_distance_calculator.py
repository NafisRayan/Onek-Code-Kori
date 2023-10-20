import cv2
import cv2.aruco as aruco

#*use this for real time (camera)
cap = cv2.VideoCapture(0)

#*use this for video clips
#cap = cv2.VideoCapture('fil/1.mp4')

# in inches
known_width = 8

#estimate
focal_length = 600

while True:
    ret, frame = cap.read()

    # Detect ArUco markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)


    for i in range(len(corners)):
        # Calculate the distance of the ArUco marker from the camera
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, known_width, focal_length, parameters=parameters)
        distance = tvec[0][0][2]
        print(f"Id: {ids} & distance from camera: {distance}")

        cv2.putText(frame, f"Distance: {distance:.2f} inches", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.putText(frame, f'Press (q) to EXIT', (20, frame.shape[0]-60), cv2.FONT_ITALIC, 0.7, (0, 250, 255), 2)
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()