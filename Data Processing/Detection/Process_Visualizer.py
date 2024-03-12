import time
from Component.SessionManager import Session
from DataManager.Frame import Frame
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath, Vec3, Camera, Lens, WindowProperties, TransparencyAttrib
import Component.Utility as Utility
import numpy as np
from direct.task import Task
import open3d as o3d
from Detection.Pose import DetectPose



class visualizer(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.point_cloud_obj: list[NodePath] = []
        self.point_anim_range: list[tuple] = []
        self.point = self.loader.loadModel("Assets/Point.stl")
        self.scene = self.loader.loadModel("models/environment")
        self.anim_progress = 0
        self.anim_end = 240
        self.last_point_start_delay = 2
        self.frame_rate = 60
        self.point_anim_time = 0.5
        self.lod = 1
        #self.scene.reparentTo(self.render)
        self.frame_data()
        self.create_point_cloud()
        self.disableMouse()
        self.setBackgroundColor(1,1,1)
        props = WindowProperties()
        props.setSize(1920,1080)
        self.win.requestProperties(props)
        
        
    
    def frame_data(self):
        trial_session = Session("data/Session_1706868542", immediate_detection=True)
        trial_session.try_add_device(168102881)
        trial_session.new_frame(1706868625406, 168102881)
        frame = trial_session.devices[trial_session.main_device].all_frames[1706868625406]
        self.abimage = frame.ab_image
        self.posedetector = DetectPose()
        self.posedetector(self.abimage)
        self.point_cloud = frame.point_cloud
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(self.point_cloud)
        o3d_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.point_cloud_normals = np.asarray(o3d_point_cloud.normals)
        self.camera_position = frame.pose[0]
        camera_orientation = frame.pose[1]
        cv2.imshow("abimage", self.abimage)
        cv2.waitKey(10)
        camera_rotation = Utility.calc_angle(camera_orientation)
        self.camera_h_midval = camera_rotation[0]/3.14159*180+90+1
        self.camera_p = -camera_rotation[1]/3.14159*180
        
        
        self.taskMgr.add(self.anim_Task,"Animation task")
        
        print(camera_rotation)
    
    def anim_Task(self, task):
        rot_fov_single_side = 10
        point_cloud_move_factor = 1
        if self.anim_progress < self.anim_end:
            if self.anim_progress == 0:
                self.camera.setPos(-self.camera_position[0],-self.camera_position[2]-6,self.camera_position[1])
                self.camera.setP(self.camera_p)
            camera_h = self.camera_h_midval - ((self.anim_progress - self.anim_end)/self.anim_end)**3 * rot_fov_single_side
            self.camera.setH(camera_h)
            for i in range(0, int(len(self.point_cloud)/self.lod)):
                coordinate = i*self.lod
                y = int(coordinate/320)
                x = coordinate - y*320
                color = self.abimage[y][x][0] / 256
                if self.anim_progress < self.point_anim_range[i][0]: 
                    x_anim = 0
                elif self.anim_progress > self.point_anim_range[i][1]:
                    x_anim = 1
                else:
                    x_anim = (self.anim_progress - self.point_anim_range[i][0])/self.point_anim_time/self.frame_rate
                point_progress_factor = (1-x_anim)**3
                opacity_factor = x_anim
                color_adj = (1 - color)*(1 - opacity_factor) + color
                point_cloud_pos = self.point_cloud_normals[coordinate]*point_cloud_move_factor*point_progress_factor + self.point_cloud[coordinate]
                self.point_cloud_obj[i].setPos(-point_cloud_pos[0], -point_cloud_pos[2], point_cloud_pos[1])
                self.point_cloud_obj[i].setColor(color_adj, color_adj, color_adj, x_anim)
                
            self.anim_progress += 1
            self.screenshot("cache/PointCloud.", imageComment=str(self.anim_progress))
        
        return Task.cont
            
    
    def create_point(self, position, color_gray: float):
        model = self.point.__copy__()
        model.setPos(Vec3(-position[0], -position[2], position[1]))
        model.setColor(color_gray, color_gray, color_gray)
        model.setScale(1)
        model.reparentTo(self.render)
        model.setTransparency(TransparencyAttrib.MAlpha)
        self.point_cloud_obj.append(model)
        
        
    def create_point_cloud(self):
        for i in range(0, int(len(self.point_cloud)/self.lod)):
            coordinate = i*self.lod
            y = int(coordinate/320)
            x = coordinate - y*320
            last_point_start_frame = self.last_point_start_delay*self.frame_rate # 120
            transposed_coordinate = x * 288 + y
            point_start_frame = (transposed_coordinate/(len(self.point_cloud)/self.lod))*(last_point_start_frame-0)
            self.point_anim_range.append((point_start_frame, point_start_frame+self.point_anim_time*self.frame_rate))
            
            point = self.point_cloud[coordinate]
            
            min = np.min(self.abimage)
            max = np.max(self.abimage)
            color = self.abimage[y][x][0]
            color_adj = (color-min)/(max-min)
            self.create_point(point, color_adj)
            
    
app = visualizer()
app.run()