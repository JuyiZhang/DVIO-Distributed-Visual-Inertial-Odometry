import math
import numpy as np
from Detection.Keypoint import Keypoint
from DataManager.Frame import Frame
import Component.Utility as Utility

keypoint_name = ["Nose","Left Eye","Right Eye","Left Ear","Right Ear","Left Shoulder","Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]


class Person:
    
    def __init__(self, keypoints: list = None, segment = None, frame: Frame = None, id = None, coordinate = None):
        if coordinate != None:
            self.coordinate = coordinate
            self.orientation = Utility.to_degree(np.arctan2(coordinate[1][2], coordinate[1][0]) + math.pi)
        else:
            self.keypoints: dict[str, Keypoint] = {}
            for i in range(0,int(len(keypoints)/2)):
                x = int(keypoints[i*2])
                y = int(keypoints[i*2+1])
                self.keypoints[keypoint_name[i]] = Keypoint((x,y), frame, segment)
            self.keypoints["Shoulder Center"] = Keypoint(self.keypoints["Left Shoulder"].get_center_coordinate_between(self.keypoints["Right Shoulder"]), frame, segment)
            self.coordinate = self.calc_coordinate()
            if len(self.coordinate) == 0:
                self.validity = False
            else:
                left_eye_kpt = np.array([keypoints[2], keypoints[3]])
                right_eye_kpt = np.array([keypoints[4], keypoints[5]])
                self.orientation = Utility.calc_angle((left_eye_kpt - right_eye_kpt), angle_type="y")
                self.id = id
                self.validity = True
        
    def calc_coordinate(self) -> np.ndarray:
        validity_priority = ["Shoulder", "Ear", "Eye"]
        
        for priority_str in validity_priority:
            if self.keypoints["Left "+priority_str].validity and self.keypoints["Right "+priority_str].validity:
                print("Use both as reference " + priority_str)
                return (self.keypoints["Left "+priority_str].coordinate_3d + self.keypoints["Right "+priority_str].coordinate_3d)/2
            elif self.keypoints["Shoulder Center"].validity:
                print("Use shoulder center as reference")
                return self.keypoints["Shoulder Center"].coordinate_3d
        
        return np.array([])
    
    def get_pos_rot(self):
        return np.append(self.coordinate, self.orientation), self.id
    
    def __str__(self):
        return "Id: " + str(self.id) + ", Position: " + str(self.coordinate)
    
    