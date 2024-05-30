import constants.folder as FOLDERCONST
import os
import cv2


class Folder:
    def init(alphabet):
        try:
            os.makedirs(f"dataset/{alphabet}")
        except:
            pass

    def file_last_index(alphabet, index):
        current_index = index
        while os.path.exists(f"dataset/{alphabet}/{index}.mp4"):
            index+1
        return current_index

    def init_file(alphabet, index):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(f"dataset/{alphabet}/{index}.mp4", fourcc, 15, (500, 500))


def show_menu(frame):
    """Show menu based on switch case

    Please update code when switch case modified
    """
    cv2.putText(frame, f"Exit: 0 || Choose Alphabet: 1 || Capture: Spacebar",
                (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
