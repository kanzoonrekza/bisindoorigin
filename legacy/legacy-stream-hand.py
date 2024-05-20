import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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

    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(1) & 0xFF

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            cv2.imshow("BISINDO-Recognition", frame)

            if key == ord('q') or key == ord('Q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
