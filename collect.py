# Utility functions
from utils.camera import init_fhd
from utils.show import Show
from utils.video import Folder
from utils.mediapipe_legacy import mp_holistic_legacy
# Library
import cv2
import numpy as np


def main():
    startCapture = False
    alphabet = None
    pTime = 0
    isSelectingAlphabet = False
    isCapturing, video_index = False, 1
    landmarks_list = []
    frame_counter, capture_length = 0, 15

    cap = init_fhd(0)
    window_name = "Dataset Collector"

    def on_mouse_click(event, x, y, flags, param):
        nonlocal startCapture
        if event == cv2.EVENT_LBUTTONDOWN and alphabet is not None:
            startCapture = True

    cap = init_fhd(1)
    with mp_holistic_legacy.setup() as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(1) & 0xFF

            if isCapturing:
                frame_counter += 1

            # Keyboard keybinds
            if key == ord('0'):
                break
            if key == ord('1'):
                alphabet = None
                isSelectingAlphabet = True
                video_index = 1
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, on_mouse_click)

            if startCapture and not isCapturing and alphabet is not None:
                isCapturing = True
                Folder.init(alphabet)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_raw = cv2.VideoWriter(
                    f'dataset/{alphabet}/{alphabet}_{video_index}_raw.mp4', fourcc, 24, (frame.shape[1], frame.shape[0]))
                out = cv2.VideoWriter(
                    f'dataset/{alphabet}/{alphabet}_{video_index}_drawed.mp4', fourcc, 24, (frame.shape[1], frame.shape[0]))
                out_np = (f'dataset/{alphabet}/{alphabet}_{video_index}.npy')
                print(f'Start Capturing {alphabet}_{video_index}')
                startCapture = False

            # Picking an alphabet
            if isSelectingAlphabet:
                Show.selecting_alphabet_notification(frame)
                if key in range(ord('a'), ord('z') + 1):
                    alphabet, isSelectingAlphabet = chr(key), False
                    video_index = Folder.get_current_index(
                        alphabet, video_index)

            if isCapturing and frame_counter < capture_length:
                out_raw.write(frame)

            # * Capture landmarks with mediapipe
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_holistic_legacy.draw(results, frame)

            if isCapturing and frame_counter < capture_length:
                mp_holistic_legacy.collectData(results, landmarks_list)
                out.write(frame)

            if isCapturing and frame_counter == capture_length:
                print(f'Capturing Done {alphabet}_{video_index}')
                out.release()
                out_raw.release()
                frame_counter = 0
                isCapturing = False
                np.save(out_np, landmarks_list)
                video_index += 1
                landmarks_list = []

            # Showing FPS and alphabet indicator
            Show.current_alphabet_and_index(frame, alphabet, video_index)
            Show.menu(frame)
            Show.current_frame_captured(
                frame, frame_counter)
            pTime = Show.fps(frame, pTime)

            cv2.imshow(window_name, frame)

    return


if __name__ == "__main__":
    main()
