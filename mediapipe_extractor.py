import os
import cv2
import mediapipe as mp
import numpy as np
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


FOLDER_NAME = 'dataset'
OUTPUT_FOLDER_NAME = 'dataset_extracted'


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

            # Extract landmarks
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
            ) if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
            ) if results.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
            ) if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
            ) if results.right_hand_landmarks else np.zeros(21*3)
            landmarks_list.append(np.concatenate([pose, face, lh, rh]))

        cap.release()

        return np.array(landmarks_list)


def main():
    for (root, folders, files) in os.walk(FOLDER_NAME):
        if root == FOLDER_NAME:
            for foldername in folders:
                try:
                    os.makedirs(f"{OUTPUT_FOLDER_NAME}/{foldername}")
                except:
                    pass
        else:
            for filename in files:
                file_path = os.path.join(os.path.relpath(
                    root, FOLDER_NAME), filename)
                if not os.path.exists(os.path.join(OUTPUT_FOLDER_NAME, os.path.splitext(file_path)[0] + "_landmarks.npy")):
                    landmarks = extract_landmarks_from_video(
                        os.path.join(FOLDER_NAME, file_path))
                    print('extracting' + os.path.join(FOLDER_NAME, file_path))
                    if landmarks is not None:
                        # if True:
                        output_file = os.path.join(OUTPUT_FOLDER_NAME, os.path.splitext(file_path)[
                            0] + "_landmarks.npy")
                        np.save(output_file, landmarks)
                        print(output_file)
    print("extraction complete")


if __name__ == "__main__":
    main()
