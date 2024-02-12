import mediapipe as mp
from Detection.Vector3 import Vector3
import numpy as np
import random
import Component.Debug as Debug
from DataManager.Frame import Frame
import Component.UDPClient as UDPClient

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

coordinate_determination_priority = [23, 11, 25, 13, 7, 27, 15] # The priority of determining coordinate, hip (0), shoulder(1), knee(1), elbow(2), head(2) ankle(2), wrist(3). If all of these data is not avai

class DetectPose:
    
    timestamp_keypoint_dict = {}
    
    def __init__(self, model_size="full", confidence = 0.5) -> None:
        options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = "model/pose_landmarker_" + model_size + ".task"),
            running_mode = VisionRunningMode.LIVE_STREAM,
            num_poses = 10,
            output_segmentation_masks = True,
            min_pose_detection_confidence = confidence,
            min_pose_presence_confidence = confidence,
            min_tracking_confidence = confidence,
            result_callback = self.process_result_and_send
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def infer(self, ab_image, timestamp: int):
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=ab_image)
        self.landmarker.detect_async(mp_image, timestamp)
    
    def __call__(self, ab_image, frame: Frame, keypoints):
        Debug.Log("Begin infer for frame" + str(frame.timestamp), category="Performance")
        self.infer(ab_image, frame.timestamp)
        self.timestamp_keypoint_dict[frame.timestamp] = [keypoints, frame]

    def process_result_and_send(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        Debug.Log("Infer completed for frame" + str(timestamp_ms), category="Performance")
        if len(result.pose_landmarks) == 0:
            Debug.Log("Person not detected", category="Detection")
            return
        
        keypoints = self.timestamp_keypoint_dict[timestamp_ms][0]
        frame_data: Frame = self.timestamp_keypoint_dict[timestamp_ms][1]
        Debug.Log("Begin matching person ID for " + str(timestamp_ms), category="Performance")   
        person_id = self.assign_person_id(result.pose_landmarks, keypoints)
        Debug.Log("Begin determine coordinate for " + str(timestamp_ms), category="Performance")   
        # Obtain 3D coordinate from depth map
        coordinate = []
        for i in range(0, len(result.pose_landmarks)): # Go over all person and obtain their 3d point
            for point_index in coordinate_determination_priority: # Go over all the coordinate
                if result.pose_landmarks[i][point_index].visibility > 0.9 and result.pose_landmarks[i][point_index + 1].visibility > 0.9: 
                    x_l = int(result.pose_landmarks[i][point_index].x * 320)
                    y_l = int(result.pose_landmarks[i][point_index].y * 288)
                    x_r = int(result.pose_landmarks[i][point_index + 1].x * 320)
                    y_r = int(result.pose_landmarks[i][point_index + 1].y * 288)
                    coordinate_l = frame_data.get_coordinate_of(x_l, y_l)
                    coordinate_r = frame_data.get_coordinate_of(x_r, y_r)
                    coordinate.append(((coordinate_l + coordinate_r)/2).tolist())
                    break
                    
                elif result.pose_landmarks[i][point_index].visibility > 0.9:
                    x_l = int(result.pose_landmarks[i][point_index].x * 320)
                    y_l = int(result.pose_landmarks[i][point_index].y * 288)
                    coordinate_l = frame_data.get_coordinate_of(x_l, y_l)
                    coordinate.append(coordinate_l.tolist()) # TODO Use predicted 3d to estimate the center
                    break
                    
                elif result.pose_landmarks[i][point_index + 1].visibility > 0.9:
                    x_r = int(result.pose_landmarks[i][point_index + 1].x * 320)
                    y_r = int(result.pose_landmarks[i][point_index + 1].y * 288)
                    coordinate_r = frame_data.get_coordinate_of(x_r, y_r)
                    coordinate.append(coordinate_r.tolist())
                    break
        Debug.Log("Begin calculate rotation vector and armature for " + str(timestamp_ms), category="Performance")      
        armature = self.armature_transform_from_world_landmark(result.pose_world_landmarks)
        Debug.Log("Sending data of frame " + str(timestamp_ms), category="Performance")
        UDPClient.send_observed_pose(person_id, coordinate, armature)   
         
        
    
    def assign_person_id(self, pose_landmarks: list, id_keypoint_list: list) -> list:
        person_id_list = []
        for person in pose_landmarks: # Go over all person detected by pose detection
            if (id_keypoint_list[1] is None): # If tracking does not work
                Debug.Log("Tracking Failed, fall back to random ID", category="0")
                person_id_list.append(random.randint(1024, 65535))
                continue
            # The list to match two run result lr shoulder, elbow, wrist, hip, knee, ankle
            joint_detect_list_pose = [*range(11,17)] + [*range(23,29)]
            joint_detect_list_keypoint = [*range(5,17)]
            person_found = False
            for i in range(0,len(id_keypoint_list[0])): # Go over all person detected by keypoint detection
                person_keypoints = id_keypoint_list[0][i] # The keypoint for the viewed person
                total_detect_distance = 0 
                outlier_count = 0
                for j in range(0,len(joint_detect_list_keypoint)): # 12 landmarks to detect: 
                    landmark_pt = person[joint_detect_list_pose[j]] # Pose detection point
                    keypoint_index = joint_detect_list_keypoint[j]*2
                    person_keypoint = np.array([person_keypoints[keypoint_index], person_keypoints[keypoint_index + 1]]) # Keypoint detection point
                    if landmark_pt.visibility < 0.9:
                        continue # Only consider the point that is visible
                    point_distance = np.linalg.norm(Vector3(landmark_pt).to_absolute() - person_keypoint)
                    if point_distance > 40:
                        outlier_count += 1
                    else:
                        total_detect_distance += point_distance
                
                Debug.Log("The total detect distance is: " + total_detect_distance.__str__() + ", there are " + outlier_count.__str__() + " outliers", "Detection")
                if total_detect_distance < 40 and outlier_count < 3: # Allow for detect distance to vary by 40 pixels or less than 3 outliers 
                    person_found = True
                    print(id_keypoint_list[1])
                    print(i)
                    person_id_list.append(id_keypoint_list[1][i])
                    
            if not person_found:
                Debug.Log("Matching unsuccessful, fall back to random id", "Detection")
                person_id_list.append(random.randint(1024, 65535))
        return person_id_list
                    
    
    def armature_transform_from_world_landmark(self, pose_world_landmarks: list):
        result = []
        
        for i in range(0,len(pose_world_landmarks)):
            person = pose_world_landmarks[i]
            
            left_shoulder = Vector3(person[11])
            right_shoulder = Vector3(person[12])
            left_elbow = Vector3(person[13])
            right_elbow = Vector3(person[14])
            left_wrist = Vector3(person[15])
            right_wrist = Vector3(person[16])
            left_hip = Vector3(person[23])
            right_hip = Vector3(person[24])
            left_knee = Vector3(person[25])
            right_knee = Vector3(person[26])
            left_ankle = Vector3(person[27])
            right_ankle = Vector3(person[28])
            neck = (right_shoulder + left_shoulder)/2
            head = (Vector3(person[7]) + Vector3(person[8]))/2
            
            body_angle = (right_hip - left_hip).angle()
            neck_angle = (head - neck).angle()
            spine_angle = (neck - (right_hip + left_hip)/2).angle()
            left_shoulder_angle = (left_elbow - left_shoulder).angle()
            right_shoulder_angle = (right_elbow - right_shoulder).angle()
            left_elbow_angle = (left_wrist - left_elbow).angle()
            right_elbow_angle = (right_wrist - right_elbow).angle()
            left_hip_angle = (left_knee - left_hip).angle()
            right_hip_angle = (right_knee - right_hip).angle()
            left_knee_angle = (left_ankle - left_knee).angle()
            right_knee_angle = (right_ankle - right_knee).angle()
            
            result.append(body_angle + spine_angle + neck_angle + left_shoulder_angle + right_shoulder_angle + left_elbow_angle + right_elbow_angle + left_hip_angle + right_hip_angle + left_knee_angle + right_knee_angle)
        
        return result