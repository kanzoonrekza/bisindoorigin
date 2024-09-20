import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class mp_hands_legacy:
    def setup(min_detection_confidence=0.5,
              min_tracking_confidence=0.5,
              max_num_hands=2):
        hands = mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                               min_tracking_confidence=min_tracking_confidence,
                               max_num_hands=max_num_hands)

        return hands

    def draw(results, frame):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


class mp_holistic_legacy:
    def setup(min_detection_confidence=0.5,
              min_tracking_confidence=0.5):

        holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

        return holistic

    def draw(results, frame):
        if results:
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def collectData(results, landmarks_list):
        if results:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
            ) if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
            ) if results.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
            ) if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
            ) if results.right_hand_landmarks else np.zeros(21*3)

            landmarks_list.append(np.concatenate([pose, face, lh, rh]))

    def collectDataHandsOnly(results, landmarks_list):
        if results:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
            ) if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
            ) if results.right_hand_landmarks else np.zeros(21*3)

            landmarks_list.append(np.concatenate([lh, rh]))
