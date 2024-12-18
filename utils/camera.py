import cv2
import constants.camera as CAMCONST


def __unopenedCamHandler(cap):
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


def __setCap(width: int, height: int, index: int):
    cap = cv2.VideoCapture(index if index else CAMCONST.INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap

# TODO: Update all description to cover camera index


def init_fhd(index=None):
    cap = __setCap(height=CAMCONST.HEIGHT_FHD,
                   width=CAMCONST.WIDTH_FHD, index=index)
    __unopenedCamHandler(cap)

    return cap


def init_hd(index=None):
    cap = __setCap(height=CAMCONST.HEIGHT_HD,
                   width=CAMCONST.WIDTH_HD, index=index)
    __unopenedCamHandler(cap)

    return cap
