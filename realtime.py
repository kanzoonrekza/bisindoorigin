import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

from utils.show_fps import show_fps

from keras.models import load_model

model = load_model('action.h5')

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    return cap


def main():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    label_index = 0
    pTime = 0
    isRecording = False
    recording_duration = 2
    video_index = 1
    cap = init_camera()

    with mp_holistic.Holistic(
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break
            key = cv2.waitKey(1) & 0xFF

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
            ) if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
            ) if results.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
            ) if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
            ) if results.right_hand_landmarks else np.zeros(21*3)

            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = np.concatenate([pose, face, lh, rh])
            sequence.append(keypoints)
            sequence = sequence[-48:]

            if len(sequence) == 48:
                # print(np.array(sequence).shape)
                res = model.predict(np.expand_dims(np.array(sequence), axis=0))[0]
                # print(CLASSES[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if CLASSES[np.argmax(res)] != sentence[-1]:
                                sentence.append(CLASSES[np.argmax(res)])
                        else:
                            sentence.append(CLASSES[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                cv2.putText(frame, str(CLASSES[np.argmax(res)]), (90, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 128, 255), 3)

            # Optional: Show FPS
            pTime = show_fps(cv2, frame, pTime)

            cv2.imshow("BISINDO-Recognition", frame)

            if key == ord('q') or key == ord('Q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(len(sequence))

if __name__ == "__main__":
    main()
