import cv2


def pick_alphabet(key, frame):
    """Handle alphabet changes in program

    To use, initiate alphabet and isSelectingAlphabet in main program 
    `alphabet = None`
    `isSelectingAlphabet = False`

    Use this function
    `alphabet, isSelectingAlphabet = pick_alphabet(key, frame)`

    Don't forget to handle the `isSelectingAlphabet` state 
    """
    cv2.putText(frame, f"Press an alphabet on keyboard to select it",
                (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if key in range(ord('a'), ord('z') + 1):
        return chr(key), False
    return None, True


def show_alphabet(frame, alphabet):
    """Show alphabet on left screen

    To use, initiate pTime in main program 
    `alphabet = None`

    Then, before show frame, use this function 
    `show_alphabet(frame, alphabet)`
    """
    cv2.putText(frame, f"Selected Alphabet: {alphabet}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
