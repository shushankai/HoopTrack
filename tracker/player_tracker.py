
import supervision as sv
import sys

from ultralytics import YOLO 
from utils import (save_stub, read_stub)

class PlayerTracker:
    def __init__(self, model_path:str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        """
        use the yolo model to predict objects in each frames and get info on them 
        """
        detections = []
        batch_size = 20
        for i in range(0, len(frames), batch_size):
            detect = self.model.predict(frames[i:i+batch_size], conf=0.5)
            detections += detect
        return detections
    
    def get_object_detections(self, frames, read_from_stubs=False, stub_path=None):
        """
        we detect the frames using our model YOLO for each frames 
        which gives us detections info on each frames in a list 
        we then 
        """
        tracks = read_stub(
            read_stub_flg= read_from_stubs,
            stub_path= stub_path
        )
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
        tracks = []

        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v:k for k, v in class_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection) # this tell that we have object detections info from yolo model 

            # Now we will pass the current frame’s detections (i.e., detected bounding boxes, class IDs,
            # and confidences from YOLO) to a tracker — in this case, ByteTrack from the supervision library — so it can:
            # •	Assign unique IDs to objects (like the basketball) across frames,
            # •	Match detections between consecutive frames,
            # •	Handle occlusions or temporary missing detections,
            # •	Smooth the object’s trajectory by tracking its movement over time.
            detection_with_tracker = self.tracker.update_with_detections(detection_supervision) 
            
            # Tracks is a list of dictionary 
            tracks.append({}) 
            
            for frame_detection in detection_with_tracker:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == class_names_inv['Player']:
                    tracks[frame_num][track_id] = {"bbox":bbox} # here frame number acts as index in the list and the track_id is the key for the dict at the index 
        
        save_stub(
                stub_path= stub_path,
                object= tracks
            )
        return tracks





    