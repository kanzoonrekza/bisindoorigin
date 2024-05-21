import time
import cv2

cTime = 0

def show_fps(frame, pTime):
    """Show fps on top left corner

    To use, initiate pTime in main program 
    `pTime=0`

    Then, before show frame, use this function 
    `pTime = show_fps(frame, pTime)`
    """
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 128, 255), 3)
    return pTime