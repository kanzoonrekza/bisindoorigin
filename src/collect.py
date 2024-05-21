from utils.camera import init_fhd
from utils.show_fps import show_fps
from utils.alphabet import pick_alphabet, show_alphabet
from utils.mediapipe_legacy import mp_hands_legacy
import cv2


def main():
    cap = init_fhd()
    pTime = 0
    isSelectingAlphabet = False
    alphabet = None

    with mp_hands_legacy.setup() as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(1) & 0xFF

            # Keybind to stop the program
            if key == ord('0'):
                break

            # Keybind to pick an alphabet
            if key == ord('1'):
                isSelectingAlphabet = True

            # Picking an alphabet
            if isSelectingAlphabet:
                alphabet, isSelectingAlphabet = pick_alphabet(key, frame)

            # Hands and draw to frame process
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_hands_legacy.draw(results, frame)

            # Showing FPS and alphabet indicator
            pTime = show_fps(frame, pTime)
            show_alphabet(frame, alphabet)

            cv2.imshow("BISINDO-Recognition", frame)

    return


if __name__ == "__main__":
    main()
