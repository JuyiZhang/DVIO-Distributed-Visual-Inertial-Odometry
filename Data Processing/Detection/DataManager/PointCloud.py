import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from DataManager.Frame import Frame

import Component.Debug as Debug
import os

device = o3d.core.Device("cuda:0")

class PointCloud:
    
    point_cloud = o3d.geometry.PointCloud()
    simplified: bool = False
    
    def __init__(self, session_dir = '') -> bool: # True if point cloud exists. false if not
        #self.point_cloud_data_queue = Queue()
        #self.simplified_point_cloud_queue = Queue()
        self.crop_volume = o3d.visualization.SelectionPolygonVolume()
        self.crop_volume.orthogonal_axis = 'y'
        self.crop_volume.axis_min = -1
        self.crop_volume.axis_max = 1
        self.directory = session_dir+"/Point_Cloud.ply"
        if os.path.exists(self.directory):
            self.point_cloud = o3d.io.read_point_cloud(self.directory)
            self.simplified = True
        else:
            self.point_cloud = o3d.geometry.PointCloud()
    
    def add_point_cloud_data(self, frame: Frame, realtime_result: bool = False):
        # Adding the point cloud to the master point cloud
        Debug.Log("Add Point Cloud Data", category="PointCloud")
        add_point_cloud = o3d.geometry.PointCloud()
        add_point_cloud.points = o3d.utility.Vector3dVector(frame.point_cloud)
        
        # Will crop point cloud if realtime
        if realtime_result:
            add_point_cloud = self.crop_volume.crop_point_cloud(add_point_cloud)
        self.point_cloud += add_point_cloud
        if realtime_result:
            self.simplified = False
            self.point_cloud_simplify()
           
    
    def get_point_cloud_data_for_display(self):
        print(self.crop_volume)
        #self.point_cloud = self.crop_volume.crop_point_cloud(self.point_cloud)
        self.point_cloud_simplify()
        self.save_point_cloud_data()
        return self.point_cloud
    
    def save_point_cloud_data(self):
        o3d.io.write_point_cloud(self.directory, self.point_cloud)
    
    def remove_ceiling(self, point_cloud: o3d.geometry.PointCloud):
        return self.crop_volume.crop_point_cloud(point_cloud)
        
    def point_cloud_simplify(self):
        if self.simplified:
            return self.point_cloud, None
        else:
            point_cloud_simplified = self.point_cloud.voxel_down_sample(voxel_size=0.07)
            self.point_cloud, r_stat_outlier = point_cloud_simplified.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
            self.simplified = True
            return self.point_cloud, r_stat_outlier