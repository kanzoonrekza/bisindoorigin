import cv2

from utils.show_fps import show_fps
from utils.save_raw_video import init_output, record_video

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

OUTPUT_FILENAME = "video-1.mp4"

def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    return cap


def main():
    pTime = 0
    isRecording = False
    recording_duration = 2

    cap = init_camera()
    output = init_output(cv2, OUTPUT_FILENAME, 24,
                          CAMERA_WIDTH, CAMERA_HEIGHT)

    while True:
        success, frame = cap.read()
        if not success:
            break
        key = cv2.waitKey(1) & 0xFF

        if isRecording:
            isRecording = record_video(
                cv2, output, frame, start_time, recording_duration)

        # Optional: Show FPS
        pTime = show_fps(cv2, frame, pTime)

        cv2.imshow("BISINDO-Recognition", frame)

        if key == ord('q'):
            break

        if key == ord(' '):
            print("Start recording")
            isRecording = True
            start_time = cv2.getTickCount()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
