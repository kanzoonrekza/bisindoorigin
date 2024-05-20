import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from utils.show_fps import show_fps

model_path = '/mediapipe_models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
final_image = None

def callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global final_image
    
    hand_landmarks_list = result.hand_landmarks
    handedness_list = result.handedness
    annotated_image = np.copy(output_image.numpy_view())

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    final_image = annotated_image
    # print('hand landmarker result: {}'.format(result.hand_landmarks == []))
    # qqqqcv2.imshow("Hand Detection", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))


options = HandLandmarkerOptions(
  base_options=BaseOptions(model_asset_path='./mediapipe_models/hand_landmarker.task'),
  num_hands=2,
  running_mode=VisionRunningMode.LIVE_STREAM,
  result_callback=callback)

def init_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    return cap

def main():
    global final_image
    pTime = 0
    cap = init_camera()
    
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(1) & 0xFF

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            landmarker.detect_async(mp_image,int(time.time() * 1000))

            shown_frame = final_image if final_image is not None else frame


            pTime = show_fps(cv2, shown_frame, pTime)
            cv2.imshow("Hand Detection", shown_frame)

            if key == ord('q') or key == ord('Q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
