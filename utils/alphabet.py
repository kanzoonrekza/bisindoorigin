import cv2


def pick_alphabet(key, frame):
    if key in range(ord('a'), ord('z') + 1):
        return chr(key), False
    return None, True


def show_current(frame, alphabet, index):
    cv2.putText(frame, f"Current: {alphabet} at index: {index}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
