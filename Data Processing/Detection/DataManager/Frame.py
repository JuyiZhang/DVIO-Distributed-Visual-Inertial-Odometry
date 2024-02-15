import math
import cv2
import numpy as np
import Component.Utility as Utility

import Component.Debug as Debug

type_def = {'abimage': '_ab_image.png', 'pose': '_pose.sci', 'depth': '_depth_map.sci', 'point_cloud': '_point_cloud.sci', 'folder': ''}

# Each frame instance represent a frame received
class Frame:
    
    # Variables of Frame
    timestamp: int # The timestamp of the frame
    pose: np.ndarray
    point_cloud: np.ndarray
    depth_map: np.ndarray
    
    device: int # The ip address of incoming device
    session_folder: str
    
    
    def __init__(self, timestamp: int, device: int, session_folder: str = None, depth_data: np.ndarray = None, ab_image: np.ndarray = None, point_cloud: np.ndarray = None, pose: np.ndarray = None) -> None:
        Debug.Log("A New Frame Is Added", category="Frame")
        self.device = device
        self.timestamp = timestamp
        self.session_folder = session_folder
        if depth_data is None:
            self.point_cloud = np.load(self.get_file_name("point_cloud"))
            self.depth_map = np.load(self.get_file_name("depth"))
            self.pose = np.load(self.get_file_name("pose"))
            self.ab_image = cv2.imread(self.get_file_name("abimage"))
        else:
            self.depth_map = depth_data
            self.pose = pose
            self.ab_image = ab_image
            self.point_cloud = point_cloud
        
    # Retrieving depth in terms of 2D coordinate
    def get_depth_of(self, x, y):
        return self.depth_map[y][x]
    
    # Retrieving 3D coordinate in terms of 2D coordinate
    def get_coordinate_of(self, x, y):
        return self.point_cloud[y*320+x]
    
    # Retrieving the position of frame
    def get_position(self) -> list:
        return self.pose[0].tolist()
    
    def get_rotation_y(self) -> float:
        return Utility.to_degree(np.arctan2(self.pose[1][2], self.pose[1][0]) + math.pi)
    
    def get_file_name(self, type) -> str:
        return self.session_folder + "/" + str(self.device) + "/" + str(int(self.timestamp/1000000)) + ("" if type == "folder" else "/" + str(self.timestamp) + type_def[type]) + ("" if type == "abimage" else ".npy")
    