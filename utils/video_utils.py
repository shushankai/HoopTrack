import cv2
import os 


def read_video(video_path:str) -> list:
    # open the video using cv2
    # read the frames and append them to list 
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frames.append(frame)
    return frames


# saving video
def save_video(output_video_frames, output_video_path):

    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        filename=output_video_path, 
        fourcc=fourcc, 
        fps=24, 
        frameSize=(output_video_frames[0].shape[1], output_video_frames[0].shape[0]) # (width, height)
    ) 
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()

