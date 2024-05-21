import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


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
