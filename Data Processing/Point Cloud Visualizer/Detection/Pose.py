import mediapipe as mp
from Detection.Vector3 import Vector3
import numpy as np
import random
import Component.Debug as Debug

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

coordinate_determination_priority = [23, 11, 25, 13, 7, 27, 15] # The priority of determining coordinate, hip (0), shoulder(1), knee(1), elbow(2), head(2) ankle(2), wrist(3). If all of these data is not avai

class DetectPose:
    
    timestamp_keypoint_dict = {}
    
    def __init__(self, model_size="full", confidence = 0.20) -> None:
        options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = "model/pose_landmarker_" + model_size + ".task"),
            running_mode = VisionRunningMode.IMAGE,
            num_poses = 10,
            output_segmentation_masks = True,
            min_pose_detection_confidence = confidence,
            min_pose_presence_confidence = confidence,
            min_tracking_confidence = confidence,
            result_callback = self.process_result_and_send
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def __call__(self, ab_image, keypoints):
        
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=ab_image)
        result = self.landmarker.detect(mp_image)

        if len(result.pose_landmarks) == 0:
            Debug.Log("Person not detected")
            return

        person_id = self.assign_person_id(result.pose_landmarks, keypoints)   
        armature = self.armature_transform_from_world_landmark(result.pose_world_landmarks)
        return person_id, armature
        
    
    def assign_person_id(self, pose_landmarks: list, id_keypoint_list: list) -> list:
        person_id_list = []
        
        for person in pose_landmarks: # Go over all person detected by pose detection
            if (id_keypoint_list[1] is None): # If tracking does not work
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
            
            Debug.Log("Body Angle")
            body_angle = (right_hip - left_hip).angle()
            Debug.Log("Neck Angle")
            neck_angle = (head - neck).angle()
            Debug.Log("Spine Angle")
            spine_angle = (neck - (right_hip + left_hip)/2).angle()
            Debug.Log("Left Shoulder Angle")
            left_shoulder_angle = (left_elbow - left_shoulder).angle()
            Debug.Log("Right Shoulder Angle")
            right_shoulder_angle = (right_elbow - right_shoulder).angle()
            Debug.Log("Left Elbow Angle")
            left_elbow_angle = (left_wrist - left_elbow).angle()
            Debug.Log("Right Elbow Angle")
            right_elbow_angle = (right_wrist - right_elbow).angle()
            Debug.Log("Left Hip Angle")
            left_hip_angle = (left_knee - left_hip).angle()
            Debug.Log("Right Hip Angle")
            right_hip_angle = (right_knee - right_hip).angle()
            Debug.Log("Left Knee Angle")
            left_knee_angle = (left_ankle - left_knee).angle()
            Debug.Log("Right Knee Angle")
            right_knee_angle = (right_ankle - right_knee).angle()
            
            result.append([body_angle + spine_angle + neck_angle + left_shoulder_angle + right_shoulder_angle + left_elbow_angle + right_elbow_angle + left_hip_angle + right_hip_angle + left_knee_angle + right_knee_angle])
        
        return result