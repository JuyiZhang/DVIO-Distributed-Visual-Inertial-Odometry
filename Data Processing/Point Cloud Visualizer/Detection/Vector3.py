from scipy.spatial.transform import Rotation as R
import numpy as np
import Component.Utility as Utility

class Vector3:
    
    def __init__(self, landmark=None, x=0, y=0, z=0):
        if (landmark is None):
            self.x = x
            self.y = y
            self.z = z
        else:
            self.x = landmark.x
            self.y = landmark.y
            self.z = landmark.z
    
    def __sub__(self, o):
        return Vector3(x = self.x - o.x, y = self.y - o.y, z = self.z - o.y)
    
    def __add__(self, o):
        return Vector3(x = self.x + o.x, y = self.y + o.y, z = self.z + o.z)
    
    def __truediv__(self, o):
        return Vector3(x = self.x / o, y = self.y / o, z = self.z / o)
    
    def __mul__(self, o):
        return Vector3(x = self.x * o.x, y = self.y * o.y, z = self.z * o.z)
    
    def angle(self):
        return Utility.calc_angle([self.x, self.y, self.z])
    
    def to_absolute(self, w = 320, h = 288):
        return np.array([self.x*w, self.y*h])