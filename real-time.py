from utils.camera import init_fhd
from utils.show import Show
from utils.mediapipe_legacy import mp_holistic_legacy
import cv2
from constants.classes import ALL_CLASSES

import tensorflow as tf
from keras.models import load_model
import numpy as np
import concurrent.futures

# Load the model after setting memory growth
model = load_model('action.h5')


def predict_action(sequence):
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    return res


def main():
    # * Unchangeable initial values
    pTime = 0
    sequence = []
    actions = np.array(ALL_CLASSES)
    sentence = []
    future = None

    # * Changable initial values
    cap = init_fhd(0)
    window_name = "BISINDO-Recognition"
    fps = 0
    threshold = 0.8

    def print_prediction(result):
        predicted_class_index = np.argmax(result)
        confidence = result[predicted_class_index] * \
            100  # Convert to percentage
        predicted_action = actions[predicted_class_index]
        print(f"Predicted: {predicted_action}, Confidence: {confidence:.2f}%")

    with mp_holistic_legacy.setup() as holistic:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                key = cv2.waitKey(int(1000 / fps) if fps > 0 else 1) & 0xFF

                # Keyboard keybinds
                if key == ord('0'):
                    break

                results = holistic.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_holistic_legacy.draw(results, frame)

                # mp_holistic_legacy.collectData(results, sequence)
                mp_holistic_legacy.collectDataHandsOnly(results, sequence)
                sequence = sequence[-14:]

                if len(sequence) == 14 and (future is None or future.done()):
                    future = executor.submit(predict_action, sequence)
                    future.add_done_callback(
                        lambda f: print_prediction(f.result()))

                pTime = Show.fps(frame, pTime)

                cv2.namedWindow(window_name)
                cv2.imshow(window_name, frame)

    return


if __name__ == "__main__":
    main()
