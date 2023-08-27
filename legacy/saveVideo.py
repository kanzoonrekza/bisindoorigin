def save_video(frames, filename):
    # We need to set resolutions.
    # So, convert them from float to integer.
    frame_width = CAMERA_WIDTH
    frame_height = CAMERA_HEIGHT

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined size.
    result = cv2.VideoWriter(filename,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             24, size)

    for frame in frames:
        # Write the frame into the video file
        result.write(frame)

    # Release the video write object
    result.release()