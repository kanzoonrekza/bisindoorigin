def init_output(cv2, filename, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps,
                           (width, height))


def record_video(cv2, output, frame, start_time, video_duration):
    output.write(frame)
    if cv2.getTickCount() - start_time >= video_duration * cv2.getTickFrequency():
        output.release()
        print(f"Recording saved")
        return False
    return True
