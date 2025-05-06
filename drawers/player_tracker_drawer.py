# Any drawer class will have draw function which call the draw eclipse method which calls the utils bbox.py method to get important values to make the drawing

# this will draw for each frames 
# we have to differentiate between players
# and we have to highlight the ball holder player differently


from .utils import draw_eclipse


class PlayerTrackDrawer:
    def __init__(self):
        pass

    def draw(self, video_frames, tracks):
        output_frames_video = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]

            # annotate the player 
            for track_id, player in player_dict.items():
                frame = draw_eclipse(frame, player['bbox'], (0,0,255), track_id=track_id)

            output_frames_video.append(frame)
            
        return output_frames_video



