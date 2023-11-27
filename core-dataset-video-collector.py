import cv2
import os

from utils.show_fps import show_fps
from utils.save_raw_video import init_output, record_video

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

FOLDER_NAME = 'dataset'
CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    return cap


def init_folder(label_index):
    try:
        os.makedirs(f"{FOLDER_NAME}/{CLASSES[label_index]}")
    except:
        pass


def main():
    label_index = 0
    pTime = 0
    isRecording = False
    recording_duration = 2
    video_index = 1
    cap = init_camera()
    init_folder(label_index)

    while True:
        success, frame = cap.read()
        if not success:
            break
        key = cv2.waitKey(1) & 0xFF

        if isRecording:
            isRecording, video_index = record_video(
                cv2, output, frame, start_time, recording_duration, video_index)

        # Optional: Show FPS
        pTime = show_fps(cv2, frame, pTime)
        cv2.putText(frame, f"{CLASSES[label_index]}", (100, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (200, 200, 60), 3)
        if isRecording:
            cv2.putText(frame, f"RECORDING {video_index}", (300, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (50, 250, 60), 3)

        cv2.imshow("BISINDO-Recognition", frame)

        if key == ord('q') or key == ord('Q'):
            break

        if key == ord(' ') and not isRecording:
            while os.path.exists(f"{FOLDER_NAME}/{CLASSES[label_index]}/video{video_index}.mp4"):
                print(
                    f"The folder {FOLDER_NAME}/{CLASSES[label_index]}/video{video_index}.mp4 exists.")
                video_index += 1

            output = init_output(
                cv2, f"{FOLDER_NAME}/{CLASSES[label_index]}/video{video_index}.mp4", 24, CAMERA_WIDTH, CAMERA_HEIGHT)
            print("Start recording")
            isRecording = True
            start_time = cv2.getTickCount()

        if key == ord('<'):  # Prev label
            label_index -= 1
            video_index = 1
            init_folder(label_index)

        if key == ord('0'):  # First label
            label_index = 0
            video_index = 1

        if key == ord('>'):  # Next label
            label_index += 1
            video_index = 1
            init_folder(label_index)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
