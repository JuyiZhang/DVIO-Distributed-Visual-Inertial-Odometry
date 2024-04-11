import random
import numpy as np
from Detection.Keypoint import TrackKeypoint
from Detection.Segment import TrackSegment
from DataManager.Person import Person
from DataManager.Frame import Frame
import cv2

detection_keypoint = TrackKeypoint()
detection_segment = TrackSegment()
    
# Input: Captured frame of Hololens
# Output: Person Object
def pose_estimation(frame: Frame, enable_3d_pose = False) -> list[Person]:
    # Grab AB Image from Hololens Capture Frame
    image = frame.ab_image.copy()
    cv2.convertScaleAbs(image, image, 2, -64)
    # Segmentation and obtain result
    result_seg = detection_segment(image)
    path = result_seg.save_dir + "\\" + result_seg.path
    img_seg = detection_segment.get_person_seg(result_seg)
    if img_seg is None:
        return None, path
    
    # Tracking and obtain result
    result_kp = detection_keypoint(image)
    keypoints_list = detection_keypoint.get_2d_keypoint(result_kp)
    bounding_box_list = result_kp.boxes.xyxy.cpu()
    confidence_list = result_kp.boxes.conf.cpu()
    if (keypoints_list is None) or (bounding_box_list is None) or (keypoints_list[0] is None):
        return None, path
    
    if (keypoints_list[1] is None):
        keypoints_list[1] = []
        for i in range(0, len(keypoints_list[0])):
            keypoints_list[1].append(random.randint(65536,99999))
    
    # Post processing that creates the skeleton of each person
    person_list = []
    path = result_kp.save_dir + "\\" + result_kp.path
    for keypoints, bbox, id, confidence in zip(keypoints_list[0], bounding_box_list, keypoints_list[1], confidence_list):
        if (keypoints is None):
            return None, path
        
        # Find the segment corresponding to the person
        person_segment, person_id = correlate_bounding_box(bbox, img_seg)
        person = Person(confidence, keypoints, person_segment, frame, id)
        if person.validity:
            person_list.append(person)
        
    return remove_duplicate(person_list), path

# Input bounding box target (from currently inspecting person) and all bounding box from image segmentation
# Output the correlated segment and id
def correlate_bounding_box(bounding_box_target, bounding_box_id_seg_array):
    for i in range(0, len(bounding_box_id_seg_array)):
        bounding_box_to_inspect = bounding_box_id_seg_array[i][1]
        if np.linalg.norm(bounding_box_target - bounding_box_to_inspect) < 30: # we regard this as same rectangle
            return bounding_box_id_seg_array[i][0], bounding_box_id_seg_array[i][2]
    return None, 99999

# Input the list of observed person
# Output the list of person that is not duplicate
def remove_duplicate(person_list: list[Person]):
    
    final_output_list: list[Person] = []
    for person in person_list:
        coord = person.coordinate
        duplicate_found = False
        
        # If output list is empty, add person immediately
        if len(final_output_list) == 0:
            final_output_list.append(person)
            continue

        else:
            for out_person in final_output_list:
                out_coord = out_person.coordinate
            
                # If two coordinate are too close, determine as duplicate
                if (np.linalg.norm(out_coord - coord) < 0.1):
                    duplicate_found = True
                    
        if not(duplicate_found):
            final_output_list.append(person)
            
    return final_output_list







