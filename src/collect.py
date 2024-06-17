from utils.camera import init_fhd
from utils.show import Show
from utils.video import Folder
from utils.mediapipe_legacy import mp_holistic_legacy
import cv2
import copy
import numpy as np


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left mouse button clicked at", x, y)


def main():
    pTime = 0
    alphabet, isSelectingAlphabet = None, False
    isCapturing, video_index = False, 1
    landmarks_list = []

    cap = init_fhd(0)
    window_name = "BISINDO-Recognition"
    fps = 0
    frame_counter, capture_length, delay_length = 0, 15, 5

    with mp_holistic_legacy.setup() as holistic:
        while True:
            success, frame = cap.read()
            scene = copy.deepcopy(frame)
            if not success:
                break

            key = cv2.waitKey(int(1000 / fps) if fps > 0 else 1) & 0xFF

            # Keybind to stop the program
            if key == ord('0'):
                break

            # Keybind to pick an alphabet
            if key == ord('1'):
                isSelectingAlphabet = True
                video_index = 1

            # Picking an alphabet
            if isSelectingAlphabet:
                Show.selecting_alphabet_notification(frame)
                if key in range(ord('a'), ord('z') + 1):
                    alphabet, isSelectingAlphabet = chr(key), False
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
                out_np = (f'dataset/{alphabet}/{alphabet}_{video_index}.npy')

            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_holistic_legacy.draw(results, frame)

            if isCapturing:
                frame_counter += 1
                if frame_counter < delay_length:
                    Show.capture_countdown(frame, frame_counter, delay_length)
                elif frame_counter == capture_length+delay_length:
                    out_raw.release()
                    out.release()
                    frame_counter = 0
                    isCapturing = False
                    np.save(out_np, landmarks_list)
                else:
                    out_raw.write(scene)
                    out.write(frame)

            # Showing FPS and alphabet indicator
            Show.current_alphabet_and_index(frame, alphabet, video_index)
            Show.menu(frame)
            Show.current_frame_captured(
                frame, frame_counter-delay_length)
            pTime = Show.fps(frame, pTime)

            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, on_mouse_click)
            cv2.imshow(window_name, frame)

    return


if __name__ == "__main__":
    main()
