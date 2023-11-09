from utils.save_raw_video import init_output
import os
import cv2
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


FOLDER_NAME = 'dataset'
OUTPUT_FOLDER_NAME = 'dataset_extracted/posehand/landmarks'
OUTPUT_VIDEO_FOLDER_NAME = 'dataset_extracted/posehand/videos'


def extract_landmarks_from_video(file_path):
    cap = cv2.VideoCapture(os.path.join(FOLDER_NAME, file_path))

    output = init_output(
        cv2, os.path.join(OUTPUT_VIDEO_FOLDER_NAME, os.path.splitext(file_path)[0] + "_extracted.mp4"), 24, 1920, 1080)

    with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            landmarks_list = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame and get the holistic results
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                resultspose = pose.process(image)
                resultshand = hands.process(image)
                # TODO: save the video with extracted landmarks

                # Extract landmarks
                pose_landmarks = np.array([[res.x, res.y, res.z] for res in resultspose.pose_landmarks.landmark]).flatten(
                ) if resultspose.pose_landmarks else np.zeros(33*4)

                if resultshand.multi_hand_landmarks:
                    for landmarks in resultshand.multi_hand_landmarks:
                        # Draw landmarks on the frame
                        mp_drawing.draw_landmarks(
                            frame, landmarks, mp_hands.HAND_CONNECTIONS)
                if resultspose.pose_landmarks:
                    # Draw pose landmarks on the frame
                    mp_drawing.draw_landmarks(frame, resultspose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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
