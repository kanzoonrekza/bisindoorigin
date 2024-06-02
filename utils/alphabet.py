import cv2


def pick_alphabet(key, frame):
    """Handle alphabet changes in program

    To use, initiate alphabet and isSelectingAlphabet in main program 
    `alphabet, isSelectingAlphabet = None, False`
    
    Use this function
    `alphabet, isSelectingAlphabet = pick_alphabet(key, frame)`

    Don't forget to handle the `isSelectingAlphabet` state 
    """
    if key in range(ord('a'), ord('z') + 1):
        return chr(key), False
    return None, True


def show_current(frame, alphabet, index):
    """Show current alphabet and index on left screen

    To use, initiate pTime in main program 
    `alphabet = None`

    Then, before show frame, use this function 
    `show_alphabet(frame, alphabet)`
    """
    cv2.putText(frame, f"Current: {alphabet} at index: {index}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
