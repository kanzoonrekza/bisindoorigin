# Utility functions

from utils.camera import init_fhd
from utils.show import Show
from utils.mediapipe_legacy import mp_holistic_legacy
from constants.classes import ALL_CLASSES

import cv2
import numpy as np
import concurrent.futures

from keras.models import load_model


def main():
    model_dir = 'model/h2-9.h5'
    model = load_model(model_dir)

    start_capture = False
    is_capturing = False

    pTime = 0
    actions = np.array(ALL_CLASSES)
    sequence = []
    future = None
    printed_result = "No Result"
    confidence_result = "-"

    cap = init_fhd(0)
    window_name = "BISINDO-Recognition"

    def predict_action(sequence):
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        return res

    def print_prediction(result):
        nonlocal confidence_result
        # Get indices of top 3 predictions
        top_indices = np.argsort(result)[-3:][::-1]
        top_actions = [actions[i] for i in top_indices]
        # Convert to percentages
        top_confidences = [result[i] * 100 for i in top_indices]

        if (top_confidences[0] > 70):
            confidence_result = str(actions[top_indices[0]])
        else:
            confidence_result = "-"

        return " ; ".join(
            [f"{action}: {confidence:.2f}%" for action,
                confidence in zip(top_actions, top_confidences)]
        )

    def update_printed_result(result):
        nonlocal printed_result
        printed_result = print_prediction(result)

    with mp_holistic_legacy.setup() as holistic:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                cv2.namedWindow(window_name)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('0'):
                    break

                if key == ord(' '):
                    if is_capturing:
                        is_capturing = False
                        out.release()
                    else:
                        start_capture = True

                if start_capture:
                    is_capturing = True

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(
                        f'realtimetest-{model_dir[6:-3]}.mp4', fourcc, 24, (frame.shape[1], frame.shape[0]))
                    start_capture = False

                results = holistic.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if model_dir.startswith('model/h'):
                    mp_holistic_legacy.collectDataHandsOnly(results, sequence)
                else:
                    mp_holistic_legacy.collectData(results, sequence)
                sequence = sequence[-14:]

                if len(sequence) == 14 and (future is None or future.done()):
                    future = executor.submit(predict_action, sequence)
                    future.add_done_callback(
                        lambda f: update_printed_result(f.result()))

                mp_holistic_legacy.draw(results, frame)
                pTime = Show.fps(frame, pTime)
                cv2.putText(frame, printed_result, (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
                cv2.putText(frame, confidence_result, (10, 160),
                            cv2.FONT_HERSHEY_PLAIN, 3, (50, 0, 0), 3)
                cv2.putText(frame, str(model_dir), (1500, 110),
                            cv2.FONT_HERSHEY_PLAIN, 3, (102, 0, 102), 3)

                if is_capturing:
                    out.write(frame)

                cv2.putText(frame, str(is_capturing), (10, 1000),
                            cv2.FONT_HERSHEY_PLAIN, 3, (153, 120, 20), 3)

                cv2.imshow(window_name, frame)

    return


if __name__ == "__main__":
    main()
