import cv2

from utils.show_fps import show_fps

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080


def main():
    pTime = 0

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

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

        cv2.imshow("BISINDO-Recognition", frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
