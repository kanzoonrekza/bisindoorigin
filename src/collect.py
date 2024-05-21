from utils.camera import init_fhd
from utils.show_fps import show_fps
from utils.mediapipe_legacy import mp_hands_legacy
import cv2


def main():
    cap = init_fhd()
    pTime = 0

    with mp_hands_legacy.setup() as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_hands_legacy.draw(results, frame)

            pTime = show_fps(frame, pTime)

            cv2.imshow("BISINDO-Recognition", frame)

    return


if __name__ == "__main__":
    main()
