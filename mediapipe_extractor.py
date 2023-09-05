import os
import cv2
import mediapipe as mp
import numpy as np
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


FOLDER_NAME = 'dataset'


def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        landmarks_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame and get the holistic results
            results = holistic.process(frame)

            # Extract landmarks if found
            if results.pose_landmarks:
                landmarks = np.array([(lm.x, lm.y, lm.z)
                                     for lm in results.pose_landmarks.landmark])
                landmarks_list.append(landmarks)

        cap.release()

        return np.array(landmarks_list)


def main():
    for foldername, subfolders, filenames in os.walk(FOLDER_NAME):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            landmarks = extract_landmarks_from_video(file_path)
            print(file_path+" being extracted")
            if landmarks is not None:
                # Save the landmarks as an array file (e.g., using pickle)
                output_file = os.path.splitext(file_path)[0] + "_landmarks.npy"
                np.save(output_file, landmarks)
    print("extraction complete")


if __name__ == "__main__":
    main()
