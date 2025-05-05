class PlayerTrackDrawer:
    def __init__(self);
        pass

    def draw(self, video_frames, tracks):
        output_frames_video = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]

            for track_id, player in player_dict.items():
                frame = draw_eclipse(frame, player['bbox'], (0,0,255))

                output_frames_video.append(frame)


            
