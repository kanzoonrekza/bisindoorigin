import cv2
import constants.camera as CAMCONST


def __unopenedCamHandler(cap):
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


def __setCap(width: int, height: int):
    cap = cv2.VideoCapture(CAMCONST.INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def init_fhd():
    """Initialize cap with FHD resolution (1920 x 1080) and handling when cap is not opened
    Example Usage:
        from utils.camera import init_fhd
        cap = init_fhd() 
    """
    cap = __setCap(height=CAMCONST.HEIGHT_FHD, width=CAMCONST.WIDTH_FHD)
    __unopenedCamHandler(cap)

    return cap


def init_hd():
    """Initialize cap with HD resolution (1280 x 720) and handling when cap is not opened
    Example Usage:
        from utils.camera import init_hd
        cap = init_hd() 
    """
    cap = __setCap(height=CAMCONST.HEIGHT_HD, width=CAMCONST.WIDTH_HD)
    __unopenedCamHandler(cap)

    return cap
