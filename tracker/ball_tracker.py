import supervision as sv
import numpy as np 
import pandas as pd 
import sys
sys.path.append('../')

from ultralytics import YOLO
from utils import (save_stub, read_stub)



class BallTracker:
    def __init__(self, model_path:str):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        detections = []
        batch_size = 20
        for i in range(0, len(frames), batch_size):
            detect = self.model.predict(frames[i:i+batch_size], conf=0.5)
            detections +=detect
        return detections
    
    def get_object_detections(self, frames, read_from_stubs, stub_path=None):
        """
        get the frames detected and then we pass those detections to our tracker 
        we save the tracking info to a dict for only cls -> ball
        """
        
        tracks = read_stub(read_stub_flg=read_from_stubs, stub_path=stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
        tracks = []
        detections= self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_names_inv = {v:k for k, v in cls_name.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence= 0 

            for frame_detect in detection_supervision:
                bbox = frame_detect[0].tolist()
                cls_id = frame_detect[3]
                confidence = frame_detect[2]
                

                if cls_id == cls_names_inv['Ball']:
                    if max_confidence<confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

                if chosen_bbox is not None:
                    tracks[frame_num][1] = {'bbox':chosen_bbox}
                

        save_stub(stub_path=stub_path,object=tracks)
        return tracks



    def remove_wrong_detections(self, ball_positions):
        max_allowed_distance = 25 # maximum allowed difference in distance of ball in two frames 
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            # frame_num = i , get(1, {}) = track_id, get(1, []) = bbox
            # tracks = #TODO structure of the tracks 
            current_bbox = ball_positions[1].get(1, {}).get(1, []) 

            if len(current_bbox) == 0:
                continue

            if last_good_frame_index == -1: 
                # First valid detection
                last_good_frame_index = i 
                continue
            
            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get(1, []) 
            frame_gap = i - last_good_frame_index
            adjusted_distance = max_allowed_distance * frame_gap

            # calculate the distance between last bbox and the current position
            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_bbox[:2])) > adjusted_distance:
                ball_positions[i] = {}
            else: 
                last_good_frame_index = i

        return ball_positions


    def interpolate_ball_positions(self, ball_positions):

        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Missing values will be filled using the interpolate - linear and backfill 
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
            
