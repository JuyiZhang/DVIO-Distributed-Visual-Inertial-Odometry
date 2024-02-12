import random
import sys
import numpy as np
from ultralytics import YOLO
import Component.Debug as Debug
import torch
from ultralytics.engine.results import Results

track_history = []

    
class TrackSegment:
    def __init__(self, model="Model/yolov8n-seg.pt"):
        m1_gpu_availability = torch.backends.mps.is_available()
        nv_gpu_availability = torch.backends.cudnn.is_available()
        if m1_gpu_availability:
            Debug.Log("The M-Series GPU is available, Allocating Resources...")
        elif nv_gpu_availability:
            Debug.Log("The Nvidia Cudnn is available, Allocating Resources...")
        else:
            Debug.Log("The GPU is not available, falling back to CPU/Using default torch device")
        self.load_model(model)
    
    def load_model(self, model_type: str):
        if not model_type.split('-')[-1] == 'seg.pt':
            sys.exit('Model not yolov8 seg')
        self.model = YOLO(model=model_type)
        
    def __call__(self, image: np.array) -> Results:
        return self.model.track(image, save=True, persist=True)[0]
    
    def get_person_seg(self, result: Results) -> list:
        boundingBoxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        if len(boundingBoxes) == 0:
            return None
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        segmentation_contours_idx = []
        for seg in result.masks.xy:
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)
        segment_person = []
        if (result[0].boxes.id == None):
            for cls, bbox, seg in zip(classes, boundingBoxes, segmentation_contours_idx):
                if str(cls) == "0":
                    segment_person.append([seg, bbox, 65535+random.randint(0,65535)])
        else:
            tracking_ids = np.array(result.boxes.id.cpu().tolist())
            for cls, bbox, seg, id in zip(classes, boundingBoxes, segmentation_contours_idx, tracking_ids):
                if str(cls) == "0":
                    segment_person.append([seg, bbox, id])
        if len(segment_person) == 0:
            return None
        return segment_person
    