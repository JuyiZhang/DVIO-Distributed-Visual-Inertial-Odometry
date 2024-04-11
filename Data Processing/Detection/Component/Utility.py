import os
import random
import math

import numpy as np

from scipy.spatial.transform import Rotation as R
import open3d as o3d
import ipaddress

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

def vector_angle(vector1: np.ndarray, vector2: np.ndarray):
    return np.arctan2(vector1[2], vector1[0]) - np.arctan2(vector2[2], vector2[0])

def get_2d_distance(vector1, vector2):
    return math.sqrt((vector1[0] - vector2[0])**2 + (vector1[1] - vector2[1])**2)

def get_3d_distance(vector1, vector2, ignore_y = True):
    return math.sqrt((vector1[0] - vector2[0])**2 + (vector1[2] - vector2[2])**2)

def rotation_matrix(rot_angle_y: float):
    return R.from_rotvec([0,rot_angle_y,0]).as_matrix()

def fast_registration(point_cloud_target, point_cloud_subject):
    fpfh_search_radius = 0.5
    distance_threshold = fpfh_search_radius/10
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(point_cloud_target, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_search_radius, max_nn=100))
    subject_fpfh = o3d.pipelines.registration.compute_fpfh_feature(point_cloud_subject, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_search_radius, max_nn=100))
    registration_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(point_cloud_subject, point_cloud_target, subject_fpfh, target_fpfh, o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return registration_result.transformation

def get_offset(transformation_matrix):
    transformation = np.array(transformation_matrix)
    rotation = R.from_matrix(transformation[0:3,0:3]).as_rotvec(True)
    translation = transformation[0:3,3]
    return rotation, translation

def calc_angle(vector, angle_type="yz"):
    
    if angle_type == "yz": # Follow unity's convention on local coordinate
        x = vector[0]
        y = vector[1]
        z = vector[2]
        print(z)
        print(x)
        rot_angle_y = np.arctan2(-z,x)
        xz = np.linalg.norm((x,z))
        rot_angle_x = np.arctan2(y, xz)
        return (rot_angle_y, rot_angle_x)
    elif angle_type == "y":
        return np.arctan2(vector[1], vector[0]) # Right Landmark -> Left Landmark
    # If the observee faces toward user to its left (right for observer), Right->Left (Left - right) generate positive result exactly cooresponding to its rotation angle

def get_inv_transformation_of_point(point: np.ndarray, origin_local_position: np.ndarray = None, origin_local_rotation: np.ndarray = None, adj_rot = 0, adj_x = 0, adj_z = 0):
    point_moved = point # + origin_local_position
    origin_local_rotation_y = 0 #-origin_local_rotation[1]
    rot_matrix = R.from_rotvec([0, origin_local_rotation_y + adj_rot, 0], True).as_matrix()
    point_rotated = np.matmul(rot_matrix, point_moved)
    #disp_rotated = np.matmul(rot_matrix, )
    #print(disp_rotated)
    return point_rotated + [adj_x,0,adj_z]

def list_all_device_timestamp(session_folder: str) -> dict[int, list[int]]:
    if session_folder.split("/")[-1].split("_")[0] != "Session":
        return None
    all_device = os.listdir(session_folder)
    all_device.sort()
    device_timestamp_dict = {}
    for device in all_device:
        if "Point_Cloud" in device or device == ".DS_Store":
            continue
        all_folder = os.listdir(session_folder + "/" + device)
        all_folder.sort()
        timestamp_list = []
        for time_big_folder in all_folder:
            if time_big_folder == ".DS_Store":
                continue
            all_file = os.listdir(session_folder + "/" + device + "/" + time_big_folder)
            all_file.sort()
            for file_name in all_file:
                timestamp = int(file_name.split("_")[0])
                if "master_pose" in file_name:
                    os.rename(session_folder + "/" + device + "/" + time_big_folder + "/" + str(timestamp) + "_master_pose.sci.npy",session_folder + "/" + device + "/" + time_big_folder + "/" + str(timestamp) + "_pose.sci.npy")
                if not(timestamp in timestamp_list):
                    timestamp_list.append(timestamp)
        device_timestamp_dict[int(device)] = timestamp_list
    return device_timestamp_dict

def save_detection_result_coordinate(filename, timestamp_dict: dict[int, bool], device_dict: dict[int, dict[int, np.ndarray]], person_coordinate_list:list[list[float]], master_pos: list[np.ndarray]):
    fp = open(filename, "w")
    flag = 0
    flag_2 = 0
    fp.write("Timestamp\t")
    for device in device_dict.keys():
        fp.write(str(ipaddress.ip_address(device)) + "x\ty\t" + "Distance to master" + "\t")
    fp.write("Detect Position x\ty\n")
    for timestamp, person_found in timestamp_dict.items():
        fp.write(str(timestamp) + "\t")
        
        #master_position_timestamp = np.array([master_pos[flag_2][0], master_pos[flag_2][2]])
        #print(master_position_timestamp)
        for device_name, coordinate in device_dict.items():
            fp.write(str(coordinate[timestamp][0]) + "\t" + str(coordinate[timestamp][1]) + "\t")
            #fp.write(str(get_2d_distance(coordinate[timestamp], master_position_timestamp)) + "\t")
        flag_2 += 1
        if person_found:  
            fp.write(str(person_coordinate_list[flag][0]) + "\t" + str(person_coordinate_list[flag][1]) + "\n")
            flag += 1
        else:
            fp.write("\n")
    fp.close()
    

def get_pose_from_timestamp(session_folder: str, device: int, timestamp: int) -> np.ndarray:
    coordinate = np.load(session_folder + "/" + str(device) + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_pose.sci.npy")
    #coordinate[0] = get_inv_transformation_of_point(coordinate[0], coordinate[2], coordinate[3])
    return coordinate

def get_ab_image_from_timestamp(session_folder: str, device: int, timestamp: int):
    return session_folder + "/" + str(device) + "/" + str(int(timestamp/1000000)) + "/" + str(timestamp) + "_ab_image.png"

def get_nearest_point_dist_sq(x: float, y: float, path: list[list[float]]) -> float:
    min_dist = 65535
    for x_path, y_path in zip(path[0],path[1]):
            dist = (x - x_path)**2 + (y - y_path)**2
            if dist < min_dist:
                min_dist = dist
    return min_dist
    
def get_path_rmse(path_predict: list[list[float]], path_actual: list[list[float]]) -> float:
    pt_count = len(path_predict[0])
    sum_error = 0
    for x, y in zip(path_predict[0], path_predict[1]):
        sum_error += get_nearest_point_dist_sq(x, y, path_actual)
    return math.sqrt(sum_error/pt_count)
    
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