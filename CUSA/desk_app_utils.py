import tkinter as tk
from tkinter import simpledialog,filedialog,messagebox,ttk
import cv2
import numpy as np
import pandas as pd
import time



#this takes in the frame and makes a bbox where the face is detected and then blurs the pixels within the box and returns then returns the entire image
def blur_face(image_rgb,frame,face_detection):
    results = face_detection.process(image_rgb)
    if results.detections:
        for detection in results.detections:
            bboxc = detection.location_data.relative_bounding_box
            ih,iw,_ = frame.shape
            x,y,w,h = int(bboxc.xmin*iw), int(bboxc.ymin*ih),int(bboxc.width*iw),int(bboxc.height*ih)
            mask = np.zeros((ih,iw),dtype=np.uint8)
            center = (x+w//2,y+h//2)
            radius = max(w,h)//2 + 5
            axes = (w//2,h//2)
            cv2.ellipse(mask,center,axes,0,0,360,(255),thickness=-1)

            blurred_frame = cv2.GaussianBlur(frame,(99,99),30)
            frame = np.where(mask[:,:,None]==255,blurred_frame,frame)
            
    return frame



#arccos of 2 points with a 3rd as a reference to get angle
def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    angle = np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0))
    return int(np.degrees(angle))



#This mostly just filters information input and output 
def get_greatest(left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle):
    dataset = pd.DataFrame.from_dict({'legs':[left_leg_angle,right_leg_angle],'upper_arms':[left_upper_arm_angle,right_upper_arm_angle],'lower_arms':[left_lower_arm_angle,right_lower_arm_angle]})
    maxes = []
    for key in dataset.keys():
        maxes.append(dataset[key].max())
    return maxes



#Assume most joints are set at ~180 degrees
def fix_angle(angle,change=180):
    return abs(angle-change)



#This just saves data neatly
def post_process(datasets):
    dfs,left_dfs,right_dfs,video_path = datasets
    dfs.index.name = 'Time(S)'

    with pd.ExcelWriter(f'{video_path}_desk_breakdown.xlsx',mode='w') as writer:  
        dfs.to_excel(writer,sheet_name='overall')
        right_dfs.to_excel(writer,sheet_name='right_side')
        left_dfs.to_excel(writer,sheet_name='left_side')



def get_desk_params(params):
    param_keys = params.keys()
    problematic_areas = ''
    for param in param_keys:
        if param == 'neck_angle' or param == 'upper_arm_angle':
            if params[param].to_list()[0] > 21:
                problematic_areas += f'{param}, '    
        else:
            if params[param].to_list()[0] < 80:
                problematic_areas += f'{param}, '
            if params[param].to_list()[0] > 100:
                problematic_areas += f'{param}, '
    problematic_areas = problematic_areas.replace('_',' ')
    return problematic_areas[:-2]



#This is where limbs are defined in relation to joints
def define_angles(mp_pose):
    #Right side
    right_upper_arm = [
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        
    ]

    right_lower_arm = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value
    ]

    right_leg = [
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value
    ]

    #Left side
    neck = [
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value
    ]

    trunk = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
    ]

    left_upper_arm = [
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        
    ]

    left_lower_arm = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value
    ]

    left_leg = [
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value
    ]

    return left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm



#Calculates the joint angles
def make_angles(landmarks,limb_sets):
    left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm = limb_sets
    neck_angle = fix_angle(calculate_angle(
                    (landmarks[neck[0]].x, landmarks[neck[0]].y),
                    (landmarks[neck[1]].x, landmarks[neck[1]].y),
                    (landmarks[neck[2]].x, landmarks[neck[2]].y)
                ),150)
                
    trunk_angle = fix_angle(calculate_angle(
        (landmarks[trunk[0]].x, landmarks[trunk[0]].y),
        (landmarks[trunk[1]].x, landmarks[trunk[1]].y),
        (landmarks[trunk[2]].x, landmarks[trunk[2]].y)
    ))

    right_upper_arm_angle = calculate_angle(
        (landmarks[right_upper_arm[0]].x, landmarks[right_upper_arm[0]].y),
        (landmarks[right_upper_arm[1]].x, landmarks[right_upper_arm[1]].y),
        (landmarks[right_upper_arm[2]].x, landmarks[right_upper_arm[2]].y)
    )
    
    right_lower_arm_angle = fix_angle(calculate_angle(
        (landmarks[right_lower_arm[0]].x, landmarks[right_lower_arm[0]].y),
        (landmarks[right_lower_arm[1]].x, landmarks[right_lower_arm[1]].y),
        (landmarks[right_lower_arm[2]].x, landmarks[right_lower_arm[2]].y)
    ))
    
    right_leg_angle = fix_angle(calculate_angle(
        (landmarks[right_leg[0]].x, landmarks[right_leg[0]].y),
        (landmarks[right_leg[1]].x, landmarks[right_leg[1]].y),
        (landmarks[right_leg[2]].x, landmarks[right_leg[2]].y)
    ))
    #Left side 
    left_upper_arm_angle = calculate_angle(
        (landmarks[left_upper_arm[0]].x, landmarks[left_upper_arm[0]].y),
        (landmarks[left_upper_arm[1]].x, landmarks[left_upper_arm[1]].y),
        (landmarks[left_upper_arm[2]].x, landmarks[left_upper_arm[2]].y)
    )
    
    left_lower_arm_angle = fix_angle(calculate_angle(
        (landmarks[left_lower_arm[0]].x, landmarks[left_lower_arm[0]].y),
        (landmarks[left_lower_arm[1]].x, landmarks[left_lower_arm[1]].y),
        (landmarks[left_lower_arm[2]].x, landmarks[left_lower_arm[2]].y)
    ))
    
    left_leg_angle = fix_angle(calculate_angle(
        (landmarks[left_leg[0]].x, landmarks[left_leg[0]].y),
        (landmarks[left_leg[1]].x, landmarks[left_leg[1]].y),
        (landmarks[left_leg[2]].x, landmarks[left_leg[2]].y)
    ))
    return left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle,trunk_angle,neck_angle
