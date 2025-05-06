import supervision as sv
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
                track_id = frame_detect[4]

                if cls_id == cls_names_inv['Ball']:
                    if max_confidence<confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

                if chosen_bbox is not None:
                    tracks[frame_num][track_id] = {'bbox':chosen_bbox}
                

        save_stub(stub_path=stub_path,object=tracks)
        return tracks

