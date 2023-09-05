import os
import time

FOLDER_NAME = 'dataset'


def main():
    for foldername, subfolders, filenames in os.walk(FOLDER_NAME):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            print(file_path)
            # time.sleep(1)


if __name__ == "__main__":
    main()
