import os
import random
import math

import numpy as np

def to_degree(radian):
    return radian / 3.14159 * 180

def euler_simplify(angle):
    angle %= 360  
    if (angle > 180):
        return angle - 360
    else:
        return angle

def generate_random_color(normalized=False):
    a = random.randint(100,255)/(255 if normalized else 1)
    b = random.randint(100,255)/(255 if normalized else 1)
    c = random.randint(100,255)/(255 if normalized else 1)
    return (a, b, c)

def get_2d_distance(vector1, vector2):
    return math.sqrt((vector1[0] - vector2[0])**2 + (vector1[1] - vector2[1])**2)

def get_3d_distance(vector1, vector2, ignore_y = True):
    return math.sqrt((vector1[0] - vector2[0])**2 + (vector1[2] - vector2[2])**2)

def calc_angle(vector, angle_type="yz"):
    
    if angle_type == "yz": # Follow unity's convention on local coordinate
        x = vector[0]
        y = vector[1]
        z = vector[2]
        rot_angle_y = np.arctan(-z,x)
        xz = np.linalg.norm((x,z))
        rot_angle_x = np.arctan(y, xz)
        return (rot_angle_y, rot_angle_x)
    elif angle_type == "y":
        return np.arctan2(vector[1], vector[0]) # Right Landmark -> Left Landmark
    # If the observee faces toward user to its left (right for observer), Right->Left (Left - right) generate positive result exactly cooresponding to its rotation angle
    
def list_all_device_timestamp(session_folder: str) -> dict[int, list[int]]:
    if session_folder.split("/")[-1].split("_")[0] != "Session":
        return None
    all_device = os.listdir(session_folder)
    device_timestamp_dict = {}
    for device in all_device:
        all_folder = os.listdir(session_folder + "/" + device)
        timestamp_list = []
        for time_big_folder in all_folder:
            all_file = os.listdir(session_folder + "/" + device + "/" + time_big_folder)
            for file_name in all_file:
                timestamp = int(file_name.split("_")[0])
                if "master_pose" in file_name:
                    os.rename(session_folder + "/" + device + "/" + time_big_folder + "/" + str(timestamp) + "_master_pose.sci.npy",session_folder + "/" + device + "/" + time_big_folder + "/" + str(timestamp) + "_pose.sci.npy")
                if not(timestamp in timestamp_list):
                    timestamp_list.append(timestamp)
        device_timestamp_dict[int(device)] = timestamp_list
    return device_timestamp_dict
    
def get_pose_from_timestamp(session_folder: str, device: int, timestamp: int) -> np.ndarray:
    return np.load(session_folder + "/" + str(device) + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_pose.sci.npy")

def get_ab_image_from_timestamp(session_folder: str, device: int, timestamp: int):
    return session_folder + "/" + str(device) + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_ab_image.png"

class Vector3D:
    def __init__(self, x = 0, y = 0, z = 0, vector = [0,0,0]) -> None:
        if (vector != [0,0,0]):
            self.x = -vector[2]
            self.y = vector[0]
            self.z = vector[1]
        elif x != 0 and y != 0 and z != 0:
            self.x = -z
            self.y = x
            self.z = y
    
    def rev_x(self):
        self.x = -self.x
        return self
    
    def vector(self):
        return [self.x, self.y, self.z]
    
    def scale(self, scale):
        self.x *= scale
        self.y *= scale
        self.z *= scale
        return self
    
    def is_zero(self):
        return True if self.x == 0 and self.y == 0 and self.z == 0 else False