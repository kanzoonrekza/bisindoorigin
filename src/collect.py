import utils.camera as CAMERA
import cv2


def main():
    cap = CAMERA.init_fhd()

    while True:
        success, frame = cap.read()
        if not success:
            break
        key = cv2.waitKey(1) & 0xFF

        cv2.imshow("Hand Detection", frame)

        if key == ord('q') or key == ord('Q'):
            break
    return


if __name__ == "__main__":
    main()
