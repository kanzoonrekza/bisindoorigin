from utils.save_raw_video import init_output
import os
import cv2
import mediapipe as mp
import numpy as np
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


FOLDER_NAME = 'dataset'
OUTPUT_FOLDER_NAME = 'dataset_extracted/holistic/landmarks'
OUTPUT_VIDEO_FOLDER_NAME = 'dataset_extracted/holistic/videos'

def extract_landmarks_from_video(file_path):
    cap = cv2.VideoCapture(os.path.join(FOLDER_NAME, file_path))

    output = init_output(
        cv2, os.path.join(OUTPUT_VIDEO_FOLDER_NAME, os.path.splitext(file_path)[0] + "_extracted.mp4"), 24, 1920, 1080)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        landmarks_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame and get the holistic results
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            # TODO: save the video with extracted landmarks

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

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            output.write(frame)

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
                try:
                    os.makedirs(f"{OUTPUT_VIDEO_FOLDER_NAME}/{foldername}")
                except:
                    pass
        else:
            for filename in files:
                file_path = os.path.join(os.path.relpath(
                    root, FOLDER_NAME), filename)
                if not os.path.exists(os.path.join(OUTPUT_FOLDER_NAME, os.path.splitext(file_path)[0] + "_landmarks.npy")):
                    landmarks = extract_landmarks_from_video(file_path)
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
