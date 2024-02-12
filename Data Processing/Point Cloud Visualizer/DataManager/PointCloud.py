import numpy as np
import open3d as o3d

from threading import Thread
from multiprocessing import Process, Queue

from scipy.spatial.transform import Rotation as R

import Component.Debug as Debug

class PointCloudManager:
    
    point_cloud_subtask_result = np.array([0,0,0])
    point_cloud_data = np.array([0,0,0])
    point_cloud_data_simplified = np.array([0,0,0])
    task_count = 0
    o3d_point_cloud = o3d.geometry.PointCloud()
    
    def __init__(self, session_dir = '', timestamp = 0) -> None:
        #self.point_cloud_data_queue = Queue()
        #self.simplified_point_cloud_queue = Queue()
        if session_dir != '':
            self.point_cloud_data = np.load(session_dir + "/Point_Cloud.sci.npy").reshape(-1,3)
            self.point_cloud_data_simplified = self.point_cloud_data
            self.o3d_point_cloud.points = o3d.utility.Vector3dVector(self.point_cloud_data.astype(np.float64))
            self.directory = session_dir
        elif timestamp != 0:
            self.directory = "data/Session_"+str(int(timestamp))
    
    def new_point_cloud_data(self, point_cloud):
        
        # Adding the point cloud to the master point cloud
        Debug.Log("Adding new point cloud")
        removed_point_cloud = self.remove_ceiling(point_cloud.reshape(-1,3))
        self.point_cloud_data = np.append(self.point_cloud_data, removed_point_cloud)
       
    def get_point_cloud_data(self):
        return self.point_cloud_data
    
    def save_point_cloud_data(self):
        np.save(self.directory + "/Point_Cloud.sci", self.point_cloud_data_simplified)
        
    def render_point_cloud(self, result: Queue):
        point_cloud_simplified = result.get()
        Debug.Log("Retrieved point cloud")
        self.point_cloud_data = np.append(self.point_cloud_data, point_cloud_simplified)
        self.task_count += 1
        if self.task_count > 5:
            result = Queue()
            Process(target=point_cloud_simplify, args=(self.point_cloud_data, result))
            Thread(target=self.render_full_point_cloud, args=(result,))
            
    def render_full_point_cloud(self, result: Queue):
        self.point_cloud_data_simplified = result.get()
        self.point_cloud_data = self.point_cloud_data_simplified
    
    def remove_ceiling(self, point_cloud_data):
        coordinate_to_remove = []
        for i in range(0,point_cloud_data.shape[0]):
            z_data = point_cloud_data[i][1]
            if z_data > 1 or z_data < -1:
                coordinate_to_remove.append(i)
        return np.delete(point_cloud_data, coordinate_to_remove, 0)
    
    def view_point_cloud(self):
        o3d.visualization.draw_geometries([self.o3d_point_cloud])
        
def point_cloud_simplify(point_cloud, result: Queue):
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float64))
    o3d_pc_simplified = o3d_pc.voxel_down_sample(voxel_size=0.07)
    o3d_pc_simplified, r = o3d_pc_simplified.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
    result.put(np.asarray(o3d_pc_simplified.points).reshape(-1,3))