import ipaddress
from threading import Thread, Event

import numpy as np
import Component.Utility as Utility
import Component.DetectionManager as DetectionManager
import Component.Debug as Debug
from DataManager.Person import Person
from DataManager.Frame import Frame
from scipy.spatial.transform import Rotation as R
from DataManager.PointCloud import PointCloud
class Device:
    
    latest_processed_image: str = ""
    latest_raw_image: np.ndarray
    rotation_offset = 0 # in degrees
    displacement_x_offset = 0 # in meter
    displacement_z_offset = 0 # in meter
    time_offset = 0
    
    def __init__(self, id: int, session_folder: str) -> None:
        self.id = id
        self.session_folder = session_folder
        self.point_cloud = PointCloud(id, session_folder)
        self.all_frames_position_device: dict[int, Person] = {}
        self.all_frames: dict[int, Frame] = {}
        self.transf_matrix: np.ndarray = None
        self.detection_flag: bool = False
        self.detect_data_ready: bool = False
        self.current_frame: Frame = None
    
    # If frame not exist, add frame, if frame exists, do nothing
    def try_add_frame(self, timestamp: int, frame: Frame = None):
        if timestamp in self.all_frames.keys():
            return
        if frame is not None:
            self.current_frame = frame
        else:
            self.current_frame = Frame(timestamp, self.id, self.session_folder)
        self.all_frames_position_device[timestamp] = Person(coordinate=self.current_frame.pose.tolist(), id=self.id)
        self.all_frames[timestamp] = self.current_frame
        self.latest_raw_image = self.current_frame.ab_image
    
    def set_offset(self, rot = 0.0, x = 0.0, z = 0.0, t = None):
        self.rotation_offset = rot
        self.displacement_x_offset = x
        self.displacement_z_offset = z
        if t is not None:
            self.time_offset = t
        Debug.Log("Set offset of device " + str(ipaddress.ip_address(self.id)) + " as " + self.offset_str())
    
    def get_point_with_offset(self, point: np.ndarray):
        return Utility.get_inv_transformation_of_point(point, adj_rot=self.rotation_offset, adj_x=self.displacement_x_offset, adj_z=self.displacement_z_offset)
    
    def offset_str(self):
        return str(self.rotation_offset) + "," + str(self.displacement_x_offset) + "," + str(self.displacement_z_offset) + "," + str(self.time_offset)
    
    def generate_point_cloud(self):
        for frame in self.all_frames.values():
            self.point_cloud.add_point_cloud_data(frame)
        return self.point_cloud.get_point_cloud_data_for_display()
    
    def detect_frame(self, timestamp: int = -1) -> list[Person]:
        if self.detection_flag:
            if timestamp == -1:
                detect_frame = self.current_frame
            else:
                detect_frame = self.all_frames[timestamp]
            detection_result, detection_image_path = DetectionManager.pose_estimation(detect_frame)
            self.detect_data_ready = True
            self.latest_processed_image = detection_image_path
            return detection_result
        else:
            return None
    
    def get_latest_processed_image(self) -> str:
        self.detect_data_ready = False
        return self.latest_processed_image
    
    def get_device_transform_at_timestamp(self, timestamp: int) -> np.ndarray:
        timestamp_frame = timestamp + self.time_offset
        prev_timestamp = 0
        transform_device_coordinate = np.array([])
        for recorded_timestamp, person in self.all_frames_position_device.items():
            if recorded_timestamp > timestamp_frame:
                recorded_timestamp_transform = np.array(person.coordinate)
                if prev_timestamp == 0:
                    transform_device_coordinate = recorded_timestamp_transform
                    break
                transform_device_coordinate = (recorded_timestamp_transform - prev_timestamp_transform) / (recorded_timestamp - prev_timestamp) * (timestamp_frame - prev_timestamp) + prev_timestamp_transform
                break
            elif recorded_timestamp == timestamp_frame:
                transform_device_coordinate = np.array(person.coordinate)
                break
            else:
                prev_timestamp_transform = np.array(person.coordinate)
                prev_timestamp = recorded_timestamp        
        
        return Utility.get_inv_transformation_of_point(transform_device_coordinate[0], transform_device_coordinate[2], transform_device_coordinate[3], self.rotation_offset, self.displacement_x_offset, -self.displacement_z_offset)
        #print(transform_device_coordinate)
        #print(np.matmul(self.get_transformation_matrix_from(transform_device_coordinate[3], -transform_device_coordinate[2]), np.array([*transform_device_coordinate[2], 1])))
        #return np.array(np.matmul(self.get_transformation_matrix_from(transform_device_coordinate[3], transform_device_coordinate[2]), np.array([*transform_device_coordinate[0], 1])))[0] # Resolve issue where np wraps the whole shit
    
    def get_transformation_matrix_from(self, rotation: np.ndarray, displacement: np.ndarray):
        rot_matrix = R.from_euler("zxy",[0, 0, -rotation[1] + self.rotation_offset],True).as_matrix()
        rot_matrix_a = np.matrix([[*rot_matrix[0], 0], [*rot_matrix[1], 0], [*rot_matrix[2], 0], [0,0,0,1]])
        disp_matrix_a = np.matrix([[1,0,0,displacement[0] + self.displacement_x_offset],[0,1,0, displacement[1]],[0,0,1, displacement[2] + self.displacement_z_offset],[0,0,0,1]])
        return np.matmul(disp_matrix_a, rot_matrix_a)
            
    def get_transformation_matrix(self):
        if self.transf_matrix is None and self.current_frame is not None:
            self.transf_matrix = self.get_transformation_matrix_from(self.current_frame.pose[3], self.current_frame.pose[2])
        elif self.current_frame is None:
            return np.identity(4)
        return self.transf_matrix
            