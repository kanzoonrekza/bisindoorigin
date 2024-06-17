import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class mp_hands_legacy:
    """Class containing ready to use mediapipe hands legacy API
    """

    def setup(min_detection_confidence=0.5,
              min_tracking_confidence=0.5,
              max_num_hands=2):
        """Setup usable mp_hands.Hands with some default preset

        To use, call `with mp_hands_legacy.setup() as hands:` to use default preset, or specify `min_detection_confidence`, `min_tracking_confidence`, and `max_num_hands` to fine-tune it.
        """
        hands = mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                               min_tracking_confidence=min_tracking_confidence,
                               max_num_hands=max_num_hands)

        return hands

    def draw(results, frame):
        """Draws results to frame.

        To use, directly call `mp_hands_legacy(results, frame)` or `mp_hands_legacy(results=results, frame=frame)`

        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


class mp_holistic_legacy:
    """Class containing ready to use mediapipe hands legacy API
    """

    def setup(min_detection_confidence=0.5,
              min_tracking_confidence=0.5):
        """Setup usable mp_holistic.Holistic with some default preset

        To use, call `with mp_holistic_legacy.setup() as holistic:` to use default preset, or specify `min_detection_confidence` and `min_tracking_confidence` to fine-tune it.
        """
        holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

        return holistic

    def draw(results, frame):
        """Draws results to frame.

        To use, directly call `mp_hands_legacy(results, frame)` or `mp_hands_legacy(results=results, frame=frame)`

        """
        if results:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def collectData(results, landmarks_list):
        """Collects data from results

        To use, call `mp_holistic_legacy.collectData(results)`

        """
        if results:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
            ) if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
            ) if results.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
            ) if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
            ) if results.right_hand_landmarks else np.zeros(21*3)

        return landmarks_list.append(np.concatenate([pose, face, lh, rh]))
