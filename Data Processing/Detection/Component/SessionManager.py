import ipaddress
import time
import threading
from multiprocessing import Pool
import os

import numpy as np

import Component.Debug as Debug

import Component.Utility as Utility
from DataManager.Person import Person
from DataManager.Device import Device
from DataManager.Frame import Frame
import Component.UDPClient as UDPClient
#from PointCloudManagerAsync import PointCloudManager



class Session:
    
    detection_flag = False # Define if a new frame is ready for detection
    detection_in_process = False # Define if a detection is in process
    observed_person: dict[int, Person] = {} # The list of observed body
    observation_history: dict[int, dict[int, Person]] = {}
    devices: dict[int, Device] = {}
    main_device = 0 # The main device for display collected data
    detected_image = np.array([]) # The image after post processing
    master_frame_updated = False
    coordinate_prediction = np.array([])
    prev_result: dict[int, Person] = {}
    prev_timestamp: int = 0
    predicted_slope: dict[int, np.ndarray] = {}
    last_successful_timestamp = 0
    
    # session_folder: The Folder of the session, use_secondary_tracking: if DVIO method is employed, immediate_detection: if detection happens immediately after adding frame
    def __init__(self, session_folder, use_secondary_tracking = True, immediate_detection = False) -> None:
        print("Creating new session with folder " + session_folder)
        self.session_folder = session_folder
        self.use_secondary_tracking = use_secondary_tracking
        self.immediate_detection = immediate_detection
        if not immediate_detection:
            threading.Thread(target=self.frame_detection, daemon=True).start()
        
    def new_frame(self, timestamp: int, device: int, frame: Frame = None):
        
        self.devices[device].try_add_frame(timestamp, frame)
            
        if (device == self.main_device and not self.immediate_detection):
            # Updating frame for primary device
            self.master_frame_updated = True
            self.detection_flag = True
        
        #UDPClient.send_all_pose(self.devices) # TODO change to update device pose
    
    def calc_position(self, timestamp: int):
        observed_person_dict = {}
        for device in self.devices.values():
            observed_person_dict[device.id] = device.detect_frame(timestamp)
        self.calculate_scene_person_position(observed_person_dict, timestamp)
    
    def detect_frame(self, device: int, timestamp: int = -1):
        if timestamp != -1:
            self.try_add_device(device)
            self.devices[device].try_add_frame(timestamp)
        observed_person_list = self.devices[device].detect_frame(timestamp)
        self.process_person_data(observed_person_list, timestamp)
    
    def try_add_device(self, device: int):
        if not(device in self.devices.keys()):
            print("Add new device: " + str(ipaddress.ip_address(device)))
            self.devices[device] = Device(device, self.session_folder)
            if self.main_device == 0:
                self.set_main_device(device)
            
    def set_device_detection(self, device: int, state = True):
        self.devices[device].detection_flag = True
    
    # Setting new main device and return the old main device address
    def set_main_device(self, device: int) -> int:
        if self.main_device != 0: 
            old_main_device = self.main_device
            self.set_device_detection(old_main_device, False)
        else:
            old_main_device = -1
        self.main_device = device
        self.set_device_detection(device, True)
        print("Init main device as: " + str(ipaddress.ip_address(device)))
        return old_main_device
    
    def frame_detection(self):
        while(True):
            if self.detection_flag and not(self.detection_in_process) and self.use_secondary_tracking:
                self.detection_in_process = True
                frame_timestamp = self.devices[self.main_device].current_frame.timestamp
                observed_person_list = self.devices[self.main_device].detect_frame()
                #UDPClient.send_observed_coordinate(observed_person_list)
                self.process_person_data(observed_person_list, frame_timestamp)
                UDPClient.send_person(observed_person_list)
                self.detection_in_process = False
                self.detection_flag = False
    
    def calculate_scene_person_position(self, observed_person_dict: dict[int, list[Person]], frame_timestamp: int):
        self.observation_history[frame_timestamp] = {}
        current_frame_person_position_observation_dict: dict[int, list[Person]] = {}
        for device_id in self.devices.keys():
            current_frame_person_position_observation_dict[device_id] = [Person(confidence=1,coordinate=self.devices[device_id].get_device_transform_at_timestamp(frame_timestamp))]
        for device_id, person_list in observed_person_dict.items():
            if person_list is not None:
                for person in person_list:
                    person_tracking_confidence = person.confidence
                    person_depth = person.depth
                    if person_depth < 0 or person_depth > 5:
                        person.confidence = 0
                        continue
                    elif person_depth > 1.4 or person_depth < 3.6:
                        depth_factor = 1
                    else:
                        depth_factor = 4/abs(2*person_depth-5)-0.8
                    device_confidence = self.devices[device_id].get_timestamp_confidence(frame_timestamp)
                    person.confidence = person_tracking_confidence * device_confidence * depth_factor * 0.25
                    person_found = False
                    for id, person_observed in current_frame_person_position_observation_dict.items():
                        if Utility.get_3d_distance(person.coordinate, person_observed.coordinate) < 0.2:
                            Debug.Log("Matching Observation Result")
                            current_frame_person_position_observation_dict[id].append(person)
                            person_found = True
                            break # Other person will not be matched
                    if not person_found:
                        Debug.Log("New Observation Result")
                        current_frame_person_position_observation_dict[person.id] = [person]
        for device_id, person_list in current_frame_person_position_observation_dict.items():
            person_position_total = np.array([0,0,0])
            person_confidence_total = 0
            for person in person_list:
                person_position_total += person.coordinate * person.confidence
                person_confidence_total += person.confidence
            if person_confidence_total > 1:
                person_confidence_total = 1
            person = Person(coordinate=person_position_total/person_confidence_total, confidence=person_confidence_total)
            if person.id in self.predicted_slope.keys():
                    
                predicted_result = self.predicted_slope[person.id] * (frame_timestamp - self.prev_timestamp) + self.prev_result[person.id].coordinate
                
                print(predicted_result)
                print(person.coordinate)
                # The data point is only regarded as valid when it does not deviate too far from prediction
                if len(self.observation_history) < 2 or Utility.get_3d_distance(predicted_result, person.coordinate) < 1.0 or frame_timestamp - self.last_successful_timestamp > 1000:
                    print("Person regarded as valid")
                    self.observation_history[frame_timestamp][person.id] = person
                    self.last_successful_timestamp = frame_timestamp
                else:
                    print("Person regarded as invalid due to deviation from prediction")
                    continue
            elif len(self.observation_history) < 3:
                self.observation_history[frame_timestamp][person.id] = person
                self.last_successful_timestamp = frame_timestamp
            
            if person.id in self.prev_result.keys():
                # Set new predicted slope based on current frame result
                self.predicted_slope[person.id] = (person.coordinate - self.prev_result[person.id].coordinate) / (frame_timestamp - self.prev_timestamp)
            
            
            # Set new previous result based on current frame result
            self.prev_result[person.id] = person
            
            # Set previous timestamp as current timestamp (note that we only set timestamp once since every person observed follows the same timestamp)
            self.prev_timestamp = frame_timestamp
            if device_id in self.devices.keys():
                self.devices[device_id].set_confidence(frame_timestamp, person_confidence_total)
        if self.observation_history[frame_timestamp] == {}:
            self.observation_history.pop(frame_timestamp)  
        
                
            
    def process_person_data(self, observed_person_list: list[Person], frame_timestamp: int):
        if observed_person_list is not None:
            self.observation_history[frame_timestamp] = {}
            # Go over all the person detected
            for person in observed_person_list:
                # Go over the list of all devices to see if we can match observed user to all the devices
                for device_id in self.devices.keys():
                    device_transform = self.devices[device_id].get_device_transform_at_timestamp(frame_timestamp)
                    if Utility.get_3d_distance(device_transform[0:3], person.coordinate) < 0.2: # If distance is smaller than 0.2
                        Debug.Log("Matching id of " + str(person.id) + " with device id " + str(device_id), category="Detection Important")
                        person.id = device_id
                        break
                # Check coordinate in line with predicted coordinate and predict next coordinate
                person.id = 0
                if person.id in self.predicted_slope.keys():
                    
                    predicted_result = self.predicted_slope[person.id] * (frame_timestamp - self.prev_timestamp) + self.prev_result[person.id].coordinate
                    
                    print(predicted_result)
                    print(person.coordinate)
                    # The data point is only regarded as valid when it does not deviate too far from prediction
                    if len(self.observation_history) < 2 or Utility.get_3d_distance(predicted_result, person.coordinate) < 1.0 or frame_timestamp - self.last_successful_timestamp > 1000:
                        print("Person regarded as valid")
                        self.observation_history[frame_timestamp][person.id] = person
                        self.last_successful_timestamp = frame_timestamp
                    else:
                        print("Person regarded as invalid due to deviation from prediction")
                        observed_person_list.remove(person)
                        break
                elif len(self.observation_history) < 3:
                    self.observation_history[frame_timestamp][person.id] = person
                    self.last_successful_timestamp = frame_timestamp
                
                if person.id in self.prev_result.keys():
                    # Set new predicted slope based on current frame result
                    self.predicted_slope[person.id] = (person.coordinate - self.prev_result[person.id].coordinate) / (frame_timestamp - self.prev_timestamp)
                
                
                # Set new previous result based on current frame result
                self.prev_result[person.id] = person
            if self.observation_history[frame_timestamp] == {}:
                self.observation_history.pop(frame_timestamp)  
            # Set previous timestamp as current timestamp (note that we only set timestamp once since every person observed follows the same timestamp)
            self.prev_timestamp = frame_timestamp
                    
                
    
    def get_point_cloud_data(self):
        return self.point_cloud_manager.get_point_cloud_data()
    
    def get_observed_list(self):
        return self.observed_list

    def get_master_pose(self):
        if self.current_frame is not None:
            return self.current_frame.get_pose()
        else:
            return np.array([])                
                
    def get_all_pose(self):
        return self.devices_frame
    
    def get_post_processed_image(self):
        if self.detected_image is not None:
            return self.detected_image
        else:
            return self.get_ab_image()
        
    def get_ab_image(self):
        if self.current_frame is not None:
            return self.current_frame.get_ab_image()
        else:
            return None
    
    def set_use_secondary_tracking(self, use_secondary_tracking = True):
        self.use_secondary_tracking = use_secondary_tracking