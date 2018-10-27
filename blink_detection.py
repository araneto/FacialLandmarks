from scipy.spatial import distance as dist
from imutils.video import VideoStream # for video from stream, or webcam
from imutils import face_utils
import imutils
import time
import dlib
import cv2

# calculate EAR (eye aspect ratio)
# https://bit.ly/2PpBmux
def ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# if ear falls bellow threshold and then rises above the threshold, we'll register a "blink"
EAR_THRESHOLD = 0.24
EAR_CONSEC_FRAMES = 5

# frame counters and total blinks
COUNTER = 0
TOTAL = 0

# dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# face_utils.FACIAL_LANDMARKS_IDXS -> return full set of facial landmarks
(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream
vs = VideoStream(src=0).start()
time.sleep(1.0)

# get video frames, resize and convert to grayscale
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    rects = detector(gray, 0)
    for face in rects:
        # convert facial landmark in 'face' to a numpy array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # extract eye coordinates, then compute ear for both eyes
        leftEye = shape[leftStart:leftEnd]
        rightEye = shape[rightStart:rightEnd]

        leftEAR = ear(leftEye)
        rightEAR = ear(rightEye)

        average_ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # if average_ear < EAR_THRESHOLD, increment frame counter COUNTER
        if average_ear < EAR_THRESHOLD:
            COUNTER += 1

        # if COUNTER (frame counter) >= EAR_CONSEC_FRAMES, increment total blinks and restart frame loop
        else:
            if COUNTER >= EAR_CONSEC_FRAMES:
                TOTAL += 1
                print("Left Eye", leftEAR)
                print("Right Eye", rightEAR)
            COUNTER = 0

        # show total blinks increasing
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(average_ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
