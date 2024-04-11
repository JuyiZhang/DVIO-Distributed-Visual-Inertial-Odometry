import math
import numpy as np
from Detection.Keypoint import Keypoint
from DataManager.Frame import Frame
import Component.Utility as Utility

keypoint_name = ["Nose","Left Eye","Right Eye","Left Ear","Right Ear","Left Shoulder","Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]


class Person:
    
    def __init__(self, confidence, keypoints: list = None, segment = None, frame: Frame = None, id = None, coordinate = None):
        if coordinate != None:
            self.coordinate = coordinate
            self.orientation = Utility.to_degree(np.arctan2(coordinate[1][2], coordinate[1][0]) + math.pi)
            self.confidence = confidence
        else:
            self.confidence = confidence
            self.keypoints: dict[str, Keypoint] = {}
            for i in range(0,int(len(keypoints)/2)):
                x = int(keypoints[i*2])
                y = int(keypoints[i*2+1])
                self.keypoints[keypoint_name[i]] = Keypoint((x,y), frame, segment)
            self.keypoints["Shoulder Center"] = Keypoint(self.keypoints["Left Shoulder"].get_center_coordinate_between(self.keypoints["Right Shoulder"]), frame, segment)
            #self.calc_kpt_validity()
            self.coordinate, self.depth = self.calc_coordinate()
            if len(self.coordinate) == 0:
                self.validity = False
            else:
                left_eye_kpt = np.array([keypoints[2], keypoints[3]])
                right_eye_kpt = np.array([keypoints[4], keypoints[5]])
                self.orientation = Utility.calc_angle((left_eye_kpt - right_eye_kpt), angle_type="y")
                self.id = id
                self.validity = True
    
    def get_bone_angle(self, kpt_1: Keypoint, kpt_2: Keypoint):
        if kpt_1.rel_3d_coordinate is not None and kpt_2.rel_3d_coordinate is not None:
            return (kpt_1.rel_3d_coordinate - kpt_2.rel_3d_coordinate).angle()
        elif kpt_1.validity and kpt_2.validity:
            return (kpt_1.Vector3() - kpt_2.Vector3()).angle()
        else:
            return None
    
    def calc_coordinate(self) -> np.ndarray:
        validity_priority = ["Shoulder", "Hip", "Eye", "Ear"]
        
        for priority_str in validity_priority:
            if self.keypoints["Left "+priority_str].validity and self.keypoints["Right "+priority_str].validity:
                return (self.keypoints["Left "+priority_str].coordinate_3d + self.keypoints["Right "+priority_str].coordinate_3d)/2, (self.keypoints["Left "+priority_str].depth + self.keypoints["Right "+priority_str].depth)/2
            elif self.keypoints["Shoulder Center"].validity:
                return self.keypoints["Shoulder Center"].coordinate_3d, self.keypoints["Shoulder Center"].depth
            
        for priority_str in validity_priority:
            if self.keypoints["Left "+priority_str].validity:
                return self.keypoints["Left "+priority_str].coordinate_3d, self.keypoints["Left "+priority_str].depth
            elif self.keypoints["Right "+priority_str].validity:
                return self.keypoints["Right "+priority_str].coordinate_3d, self.keypoints["Right "+priority_str].depth
        return np.array([]), 0
    
    def calc_kpt_validity(self):
        depth_array = []
        for keypoint in self.keypoints.values():
            depth_array.append(keypoint.depth)
            
        q1 = np.percentile(np.array(depth_array), 25) 
        q3 = np.percentile(np.array(depth_array), 75) 
        iqr = q3 - q1
        
        for kpt_name, keypoint in self.keypoints.items():
            if keypoint.depth < q1 - 1.5 * iqr or keypoint.depth > q3 + 1.5 * iqr:
                keypoint.validity = False
                print(kpt_name + " is set to invalid due to depth value: " + str(keypoint.depth))
            else:
                keypoint.validity = True
    
    def get_pos_rot(self):
        return np.append(self.coordinate, self.orientation), self.id
    
    def __str__(self):
        return "Id: " + str(self.id) + ", Position: " + str(self.coordinate)
    
    