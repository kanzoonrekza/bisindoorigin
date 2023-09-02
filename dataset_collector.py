import cv2

from utils.show_fps import show_fps

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

OUTPUT_FILENAME = "video-1.mp4"
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4


def main():
    pTime = 0

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, 15,
                          (CAMERA_WIDTH, CAMERA_HEIGHT))

    recording_started = False
    recording_duration = 2  # 2 seconds

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        success, frame = cap.read()
        if not success:
            break
        key = cv2.waitKey(1) & 0xFF

        # Optional: Show FPS 
        pTime = show_fps(cv2, frame, pTime)

        if recording_started:
            out.write(frame)
            if cv2.getTickCount() - start_time >= recording_duration * cv2.getTickFrequency():
                recording_started = False
                out.release()
                print(f"Recording saved as {OUTPUT_FILENAME}")

        cv2.imshow("BISINDO-Recognition", frame)
        if key == ord('q'):
            break
        if key == ord(' '):
            print("pressed spacebar")
            recording_started = True
            start_time = cv2.getTickCount()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
