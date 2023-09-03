import cv2
import os

from utils.show_fps import show_fps
from utils.save_raw_video import init_output, record_video

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

FOLDER_NAME = 'dataset'
CLASS_TAKEN = 'a'
BASE_PATH = f"{FOLDER_NAME}/{CLASS_TAKEN}"


def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    return cap


def init_folder():
    try:
        os.makedirs(BASE_PATH)
    except:
        pass


def main():
    init_folder()
    pTime = 0
    isRecording = False
    recording_duration = 2
    video_index = 1

    cap = init_camera()

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
        if isRecording:
            cv2.putText(frame, "RECORDING", (300, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (50, 200, 60), 3)

        cv2.imshow("BISINDO-Recognition", frame)

        if key == ord('q') or key == ord('Q'):
            break

        if key == ord(' '):
            while os.path.exists(f"{BASE_PATH}/video{video_index}.mp4"):
                print(
                    f"The folder {BASE_PATH}/video{video_index}.mp4 exists.")
                video_index += 1

            output = init_output(
                cv2, f"{BASE_PATH}/video{video_index}.mp4", 24, CAMERA_WIDTH, CAMERA_HEIGHT)
            print("Start recording")
            isRecording = True
            start_time = cv2.getTickCount()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
