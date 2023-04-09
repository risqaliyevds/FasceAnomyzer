import argparse
import cv2
import mediapipe as mp
from func import *

# Parser for command-line
args = argparse.ArgumentParser()
# Add arguments
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default= None)
# Parse
args = args.parse_args()

# Detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

    # For image arguments
    if args.mode in ['image']:
        # Read image
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)
        # Save image to Output Image file
        cv2.imwrite('./data/Output Image/blur_img1.jpg', img)

    # For video arguments
    elif args.mode in ['video']:
        # Load video to variable
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        # Get output video
        output_video = cv2.VideoWriter('data/Output Video/output2.mp4',
                                       cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))
        # Loop to blur video
        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()
        # Close video
        cap.release()
        output_video.release()

    # For webcam
    elif args.mode in ['webcam']:
        # Open webcam
        cap = cv2.VideoCapture(0)
        # Get information (arrays) fromweb cam
        ret, frame = cap.read()
        while True:
            # Blur face in webcam
            frame = process_img(frame, face_detection)
            # Show it on real time
            cv2.imshow('frame', frame)
            # Break loop
            if cv2.waitKey(1) == ord('q'):
                break
            ret, frame = cap.read()
        # Close
        cap.release()
        cv2.destroyAllWindows()