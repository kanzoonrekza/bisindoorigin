from utils.camera import init_fhd
from utils.show import Show
from utils.video import Folder
from utils.mediapipe_legacy import mp_holistic_legacy
import cv2
import numpy as np


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left mouse button clicked at", x, y)


def main():
    # * Unchangeable initial values
    pTime = 0
    alphabet, isSelectingAlphabet = None, False
    isCapturing, video_index = False, 1
    landmarks_list = []

    # * Changable initial values
    cap = init_fhd(0)
    window_name = "BISINDO-Recognition"
    fps = 0
    frame_counter, capture_length = 0, 15
    isCapturingDrawed = True

    with mp_holistic_legacy.setup() as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(int(1000 / fps) if fps > 0 else 1) & 0xFF

            # Keyboard keybinds
            if key == ord('0'):
                break
            if key == ord('1'):
                isSelectingAlphabet = True
                video_index = 1
            if key == ord(' ') and not isCapturing and alphabet is not None:
                isCapturing = True
                Folder.init(alphabet)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_raw = cv2.VideoWriter(
                    f'dataset/{alphabet}/{alphabet}_{video_index}_raw.mp4', fourcc, fps if fps > 0 else 24, (frame.shape[1], frame.shape[0]))
                if isCapturingDrawed:
                    out = cv2.VideoWriter(
                        f'dataset/{alphabet}/{alphabet}_{video_index}_drawed.mp4', fourcc, fps if fps > 0 else 24, (frame.shape[1], frame.shape[0]))
                out_np = (f'dataset/{alphabet}/{alphabet}_{video_index}.npy')
                print(f'Start Capturing {alphabet}_{video_index}')

            # Picking an alphabet
            if isSelectingAlphabet:
                Show.selecting_alphabet_notification(frame)
                if key in range(ord('a'), ord('z') + 1):
                    alphabet, isSelectingAlphabet = chr(key), False
                    video_index = Folder.get_current_index(
                        alphabet, video_index)

            if isCapturing:
                frame_counter += 1

            if isCapturing and frame_counter < capture_length:
                out_raw.write(frame)

            # * Capture landmarks with mediapipe
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_holistic_legacy.draw(results, frame)

            if isCapturing and frame_counter < capture_length:
                mp_holistic_legacy.collectData(results, landmarks_list)
                if isCapturingDrawed:
                    out.write(frame)

            if isCapturing and frame_counter == capture_length:
                print(f'Capturing Done {alphabet}_{video_index}')
                if isCapturingDrawed:
                    out.release()
                out_raw.release()
                frame_counter = 0
                isCapturing = False
                np.save(out_np, landmarks_list)
                video_index += 1
                print(f'NUMPY {landmarks_list}')

            # Showing FPS and alphabet indicator
            Show.current_alphabet_and_index(frame, alphabet, video_index)
            Show.menu(frame)
            Show.current_frame_captured(
                frame, frame_counter)
            pTime = Show.fps(frame, pTime)

            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, on_mouse_click)
            cv2.imshow(window_name, frame)

    return


if __name__ == "__main__":
    main()
