import sys
import cv2
import numpy as np
from pydantic import BaseModel
from DataManager.Frame import Frame
import Component.Debug as Debug
import ultralytics
from ultralytics.engine.results import Results

from Detection.Vector3 import Vector3

# Define keypoint
class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

class TrackKeypoint:
    def __init__(self, yolov8_model='Model/yolov8m-pose'):
        self.yolov8_model = yolov8_model
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    def __load_model(self):
        if not self.yolov8_model.split('-')[-1] == 'pose':
            sys.exit('Model not yolov8 pose')
        self.model = ultralytics.YOLO(model=self.yolov8_model)

        # extract function keypoint
    def extract_keypoint(self, keypoint: np.ndarray) -> list:
        if (len(keypoint) == 0):
            return None
        # nose
        nose_x, nose_y = keypoint[self.get_keypoint.NOSE]
        # eye
        left_eye_x, left_eye_y = keypoint[self.get_keypoint.LEFT_EYE]
        right_eye_x, right_eye_y = keypoint[self.get_keypoint.RIGHT_EYE]
        # ear
        left_ear_x, left_ear_y = keypoint[self.get_keypoint.LEFT_EAR]
        right_ear_x, right_ear_y = keypoint[self.get_keypoint.RIGHT_EAR]
        # shoulder
        left_shoulder_x, left_shoulder_y = keypoint[self.get_keypoint.LEFT_SHOULDER]
        right_shoulder_x, right_shoulder_y = keypoint[self.get_keypoint.RIGHT_SHOULDER]
        # elbow
        left_elbow_x, left_elbow_y = keypoint[self.get_keypoint.LEFT_ELBOW]
        right_elbow_x, right_elbow_y = keypoint[self.get_keypoint.RIGHT_ELBOW]
        # wrist
        left_wrist_x, left_wrist_y = keypoint[self.get_keypoint.LEFT_WRIST]
        right_wrist_x, right_wrist_y = keypoint[self.get_keypoint.RIGHT_WRIST]
        # hip
        left_hip_x, left_hip_y = keypoint[self.get_keypoint.LEFT_HIP]
        right_hip_x, right_hip_y = keypoint[self.get_keypoint.RIGHT_HIP]
        # knee
        left_knee_x, left_knee_y = keypoint[self.get_keypoint.LEFT_KNEE]
        right_knee_x, right_knee_y = keypoint[self.get_keypoint.RIGHT_KNEE]
        # ankle
        left_ankle_x, left_ankle_y = keypoint[self.get_keypoint.LEFT_ANKLE]
        right_ankle_x, right_ankle_y = keypoint[self.get_keypoint.RIGHT_ANKLE]
        
        return [
            nose_x, nose_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
            left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_shoulder_x, left_shoulder_y,
            right_shoulder_x, right_shoulder_y, left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y,
            left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, left_hip_x, left_hip_y,
            right_hip_x, right_hip_y, left_knee_x, left_knee_y, right_knee_x, right_knee_y,        
            left_ankle_x, left_ankle_y,right_ankle_x, right_ankle_y
        ]
    
    def get_2d_keypoint(self, results: Results):
        result_keypoint_list = []
        for result_keypoint in results.keypoints.xy.cpu().numpy():
            keypoint_data = self.extract_keypoint(result_keypoint)
            if (keypoint_data == None):
                return [None, None]
            result_keypoint_list.append(keypoint_data)
        result_id_list = None
        if (results.boxes.id is not None):
            result_id_list = results.boxes.id.int().cpu().tolist()
        return [result_keypoint_list, result_id_list]
    
    def __call__(self, image: np.array) -> Results:
        results: Results = self.model.track(image, save=True, persist=True)[0]
        return results
    
    def get_person_count(self, results: Results):
        return len(results.keypoints.xy.cpu().numpy())


class Keypoint:
    
    def __init__(self, coordinate_2d: tuple, frame: Frame, person_segment, rotation = None, rel_3d_coordinate: Vector3 = None) -> None:
        self.x = coordinate_2d[0] - 1
        self.y = coordinate_2d[1] - 1
        if rel_3d_coordinate is not None:
            self.rel_3d_coordinate = rel_3d_coordinate
        else:
            self.rel_3d_coordinate = None
        self.coordinate_3d = frame.get_coordinate_of(self.x, self.y)
        self.rotation = rotation
        self.validity = self.get_validity(person_segment, self.coordinate_3d, coordinate_2d, frame.get_position())
        self.depth = frame.get_depth_of(self.x, self.y)
    
    def get_validity(self, image_segment, threed_coordinate, coordinate_2d, observer_coord):
        x = coordinate_2d[0]
        y = coordinate_2d[1]
        depth_norm = np.linalg.norm(threed_coordinate - observer_coord)
        if depth_norm < 0.3 or (x <= 0 and y <= 0):
            return False
        else:
            if image_segment is None:
                Debug.Log("No valid segment is assigned, regard as invalid", category="Detection")
                return False
            if cv2.pointPolygonTest(image_segment, (x,y), False) > 0:
                Debug.Log("Passed Polygon Test", category="Detection")
                return True
            Debug.Log("Failed Polygon Test", category="Detection")
            return False
        
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "), 3D Depth: " + str(self.coordinate_3d) + ", 2D Depth: " + str(self.depth) + ", Validity:" + str(self.validity)
    
    def get_center_coordinate_between(self, other) -> tuple:
        if self.x == -1 or other.x == -1:
            return (-1,-1)
        return (int((self.x + other.x)/2), int((self.y + other.y)/2))
    
    def Vector3(self):
        return Vector3(self.coordinate_3d[0], self.coordinate_3d[1], self.coordinate_3d[2])
    
    