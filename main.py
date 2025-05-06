from utils import read_video, save_video
from tracker import PlayerTracker, BallTracker
from drawers import PlayerTrackDrawer, BallTrackDrawer

def main():
    print("Yo program started!")
    video_frames = read_video(video_path="input_videos/video_3.mp4")
    player_tracker = PlayerTracker('model/best.pt')
    ball_tracker = BallTracker('model/best.pt')


    player_tracks =  player_tracker.get_object_detections(
        frames=video_frames, 
        read_from_stubs=True, 
        stub_path= "stub/player_tracker.pkl"
    )

    ball_tracks = ball_tracker.get_object_detections(
        frames=video_frames,
        read_from_stubs=True, 
        stub_path="stub/ball_tracker.pkl"
    )

    player_drawer = PlayerTrackDrawer()
    ball_drawer = BallTrackDrawer()

    output_video_frame = player_drawer.draw(video_frames, player_tracks)
    output_video_frame = ball_drawer.draw(output_video_frame, ball_tracks)

    
    save_video(output_video_frame , "output_video/video_3.avi")

if __name__ == "__main__":
    main()
