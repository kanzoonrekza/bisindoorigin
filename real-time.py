from utils.camera import init_fhd
from utils.show import Show
from utils.mediapipe_legacy import mp_holistic_legacy
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('action.h5')


def main():
    # * Unchangeable initial values
    pTime = 0

    # * Changable initial values
    cap = init_fhd(1)
    window_name = "BISINDO-Recognition"
    fps = 0

    with mp_holistic_legacy.setup() as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(int(1000 / fps) if fps > 0 else 1) & 0xFF

            # Keyboard keybinds
            if key == ord('0'):
                break

            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_holistic_legacy.draw(results, frame)

            pTime = Show.fps(frame, pTime)

            cv2.namedWindow(window_name)
            cv2.imshow(window_name, frame)

    return


if __name__ == "__main__":
    main()
