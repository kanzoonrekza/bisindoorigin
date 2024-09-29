# Utility functions
from utils.camera import init_fhd
from utils.show import Show
from utils.mediapipe_legacy import mp_holistic_legacy
from utils.video import Folder

import cv2
import numpy as np


def main():
    pTime = 0
    start_capture = False
    is_capturing, video_index = False, 1
    frame_counter, capture_length = 0, 15
    alphabet = None
    is_selecting_alphabet = False
    landmarks_list = []

    cap = init_fhd(0)
    window_name = "Dataset Collector"

    def on_mouse_click(event, x, y, flags, param):
        nonlocal start_capture
        if event == cv2.EVENT_LBUTTONDOWN and alphabet is not None:
            start_capture = True

    with mp_holistic_legacy.setup() as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break
            cv2.namedWindow(window_name)

            key = cv2.waitKey(1) & 0xFF

            if is_capturing:
                frame_counter += 1

            if key == ord('0'):
                break
            if key == ord('1'):
                alphabet = None
                is_selecting_alphabet = True
                video_index = 1
            cv2.setMouseCallback(window_name, on_mouse_click)

            if start_capture and not is_capturing and alphabet is not None:
                is_capturing = True
                Folder.init(alphabet)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_raw = cv2.VideoWriter(
                    f'dataset/{alphabet}/{alphabet}_{video_index}_raw.mp4', fourcc, 24, (frame.shape[1], frame.shape[0]))
                out = cv2.VideoWriter(
                    f'dataset/{alphabet}/{alphabet}_{video_index}_drawed.mp4', fourcc, 24, (frame.shape[1], frame.shape[0]))
                out_np = (f'dataset/{alphabet}/{alphabet}_{video_index}.npy')
                print(f'Start Capturing {alphabet}_{video_index}')
                start_capture = False

            # Picking an alphabet
            if is_selecting_alphabet:
                Show.selecting_alphabet_notification(frame)
                if key in range(ord('a'), ord('z') + 1):
                    alphabet, is_selecting_alphabet = chr(key), False
                    video_index = Folder.get_current_index(
                        alphabet, video_index)

            if is_capturing and frame_counter < capture_length:
                out_raw.write(frame)

            # * Capture landmarks with mediapipe
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_holistic_legacy.draw(results, frame)

            if is_capturing and frame_counter < capture_length:
                mp_holistic_legacy.collectData(results, landmarks_list)
                out.write(frame)

            if is_capturing and frame_counter == capture_length:
                print(f'Capturing Done {alphabet}_{video_index}')
                out.release()
                out_raw.release()
                frame_counter = 0
                is_capturing = False
                np.save(out_np, landmarks_list)
                video_index += 1
                video_index = Folder.get_current_index(alphabet, video_index)
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
