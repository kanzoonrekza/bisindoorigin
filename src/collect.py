from utils.camera import init_fhd
from utils.show_fps import show_fps
from utils.video import Folder, show_menu
from utils.alphabet import pick_alphabet, show_current
from utils.mediapipe_legacy import mp_hands_legacy
import cv2
import copy
import os


def main():
    cap = init_fhd(0)
    pTime = 0
    fps = 15
    alphabet, isSelectingAlphabet = None, False
    isCapturing, video_index = False, 1
    frame_counter, capture_length = 0, 30

    with mp_hands_legacy.setup() as hands:
        while True:
            success, frame = cap.read()
            scene = copy.deepcopy(frame)
            if not success:
                break
            key = cv2.waitKey(int(1000 / fps)) & 0xFF

            # Keybind to stop the program
            if key == ord('0'):
                break

            # Keybind to pick an alphabet
            if key == ord('1'):
                isSelectingAlphabet = True
                video_index = 1

            # Picking an alphabet
            if isSelectingAlphabet:
                alphabet, isSelectingAlphabet = pick_alphabet(key, frame)
                if key in range(ord('a'), ord('z') + 1):
                    video_index = Folder.get_current_index(
                        alphabet, video_index)

            if key == ord(' ') and not isCapturing and alphabet is not None:
                isCapturing = True
                Folder.init(alphabet)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_raw = cv2.VideoWriter(
                    f'dataset/{alphabet}/{alphabet}_{video_index}_raw.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))
                out = cv2.VideoWriter(
                    f'dataset/{alphabet}/{alphabet}_{video_index}_drawed.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))

            # Hands and draw to frame process
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_hands_legacy.draw(results, frame)

            if isCapturing:
                print(f"capturing frame {frame_counter}")
                out_raw.write(scene)
                out.write(frame)
                frame_counter += 1

                if frame_counter == capture_length:
                    out_raw.release()
                    out.release()
                    frame_counter = 0
                    isCapturing = False
                    video_index += 1

            # Showing FPS and alphabet indicator
            show_current(frame, alphabet, video_index)
            pTime = show_fps(frame, pTime)
            show_menu(frame)

            cv2.imshow("BISINDO-Recognition", frame)

    return


if __name__ == "__main__":
    main()
