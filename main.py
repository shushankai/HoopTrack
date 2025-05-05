from utils import read_video, save_video
from tracker import PlayerTracker

def main():
    print("Yo program started!")
    frames = read_video(video_path="input_videos/video_3.mp4")
    player_tracker = PlayerTracker('model/best.pt')

    results =  player_tracker.get_object_detections(
        frames=frames, 
        read_from_stubs=True, 
        stub_path= "stub/player_tracker.pkl"
    )
    print(results)
    # save_video(frames, "output_video/video_3.avi")

if __name__ == "__main__":
    main()
