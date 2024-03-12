import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from DataManager.Frame import Frame

import Component.Debug as Debug
import os

device = o3d.core.Device("cuda:0")

class PointCloud:
    
    
    
    def __init__(self, device, session_dir = '') -> bool: # True if point cloud exists. false if not
        #self.point_cloud_data_queue = Queue()
        #self.simplified_point_cloud_queue = Queue()
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud_data = []
        self.simplified: bool = False
        
        self.crop_volume = o3d.visualization.SelectionPolygonVolume()
        self.crop_volume.orthogonal_axis = 'y'
        self.crop_volume.axis_min = -1
        self.crop_volume.axis_max = 1
        self.crop_volume.bounding_polygon =o3d.utility.Vector3dVector(np.array([[10,0,10],[10,0,-10],[-10,0,-10],[-10,0,10]]))
        #self.crop_volume.bounding_polygon = o3d.utility.Vector3dVector(np.array([[1.5,0,0],[-1.5,0,-4.5],[-3.33,0,-3.33],[-4,0,-7],[0,0,-6],[4,0,0],[0,0,4],[-2,0,2],[-6,0,-6],[-4.05,0,-7],[-3.38,0,-3.33],[0,0,2]], np.float64)) 
        #self.crop_volume.bounding_polygon = o3d.utility.Vector3dVector(np.array([[-2,0,1],[0,0,2],[2,0,-3],[2,0,-4],[4,0,-6],[6,0,-2],[2,0,5],[-5,0,2],[0,0,-6],[3.9,0,-6],[1.9,0,-4]], np.float64)) 
        
        self.directory = session_dir+"/Point_Cloud_" + str(device) + ".ply"
        if os.path.exists(self.directory):
            self.point_cloud = o3d.io.read_point_cloud(self.directory)
            self.simplified = True
        else:
            self.point_cloud = o3d.geometry.PointCloud()
    
    def add_point_cloud_data(self, frame: Frame, realtime_result: bool = False):
        # Adding the point cloud to the master point cloud
        Debug.Log("Add Point Cloud Data", category="PointCloud")
        
        
        
        # Will crop point cloud if realtime
        if realtime_result:
            add_point_cloud = o3d.geometry.PointCloud()
            add_point_cloud.points = o3d.utility.Vector3dVector(frame.point_cloud)
            add_point_cloud = self.crop_volume.crop_point_cloud(add_point_cloud)
            self.point_cloud += add_point_cloud
            self.simplified = False
            self.point_cloud_simplify()
        else:
            self.point_cloud_data.append(frame.point_cloud)
           
    
    def get_point_cloud_data_for_display(self) -> o3d.geometry.PointCloud:
        if not(os.path.exists(self.directory)):
            pcd = np.array(self.point_cloud_data).reshape(-1,3).astype(np.float64)
            self.point_cloud.points = o3d.utility.Vector3dVector(pcd)
            self.point_cloud = self.crop_volume.crop_point_cloud(self.point_cloud)
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
            point_cloud_simplified = self.point_cloud.voxel_down_sample(voxel_size=0.1)
            self.point_cloud, r_stat_outlier = point_cloud_simplified.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
            self.point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
            self.simplified = True
            return self.point_cloud, r_stat_outlier