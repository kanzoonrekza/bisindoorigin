import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from utils.show_fps import show_fps

model_path = './mediapipe_models/pose_landmarker_heavy.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
final_image = None

def callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global final_image
    print(result.pose_landmarks,'\n')
    
    pose_landmarks_list = result.pose_landmarks
    annotated_image = np.copy(output_image.numpy_view())
    print('initialized','\n')

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
        
    final_image = annotated_image
    # print('hand landmarker result: {}'.format(result.hand_landmarks == []))


options = PoseLandmarkerOptions(
  base_options=BaseOptions(model_asset_path=model_path),
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
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(1) & 0xFF

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            landmarker.detect_async(mp_image,int(time.time() * 1000))

            shown_frame = final_image if final_image is not None else frame


            pTime = show_fps(cv2, shown_frame, pTime)
            cv2.imshow("Pose Detection", shown_frame)

            if key == ord('q') or key == ord('Q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
