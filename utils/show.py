import cv2
import time
import cv2

cTime = 0


class Show:
    def menu(frame):
        cv2.putText(frame, f"Exit: 0 || Choose Alphabet: 1 || Capture: Spacebar",
                    (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)

    def selecting_alphabet_notification(frame):
        cv2.putText(frame, f"Press an alphabet on keyboard to select it",
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 255), 2)

    def current_alphabet_and_index(frame, alphabet, index):
        cv2.putText(frame, f"Current: {alphabet} at index: {index}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def fps(frame, pTime):
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 128, 255), 3)
        return pTime

    def current_frame_captured(frame, frame_counter):
        cv2.putText(frame, f"Capturing frame {frame_counter}", (10, 130),
                    cv2.FONT_HERSHEY_PLAIN, 3, (67, 128, 198), 3)

    def capture_countdown(frame, frame_counter, delay_length):
        cv2.putText(frame, f"Capturing in {delay_length - frame_counter}", (10, 130),
                    cv2.FONT_HERSHEY_PLAIN, 3, (128, 128, 25), 3)
