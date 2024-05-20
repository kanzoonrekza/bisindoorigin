import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

CAMERA_INDEX = 0
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480


def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    return cap


def main():
    cap = init_camera()

    with mp_holistic.Holistic(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(1) & 0xFF

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.imshow("BISINDO-Recognition", frame)

            if key == ord('q') or key == ord('Q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
