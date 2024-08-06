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


#Counts frequency of maximum ejms score per category
def ejms_counter(df):
    count_dict = []
    keyset = df.keys()
    for key in keyset:
        count_dict.append(df[df[key]==df[key].max()][key].count())
    overall_df = pd.DataFrame.from_dict({1:count_dict},orient='index',columns = keyset)
    return overall_df



#calculate height to pixel ratio
def calculate_pix_height(neck,ankle,person_h):
    if neck and ankle:
        pixel_height = abs(neck[1] - ankle[1])
        pixels_per_foot = pixel_height/person_h
        return pixels_per_foot 



#uses the magnitude of 2 points to get distance
def horizontal_dist(points,prev_points,pixels_per_foot):
    points = np.array(points)
    prev_points = np.array(prev_points)
    pixel_distance = np.linalg.norm(np.array(points)-np.array(prev_points))
    dist_carried = abs(pixel_distance/pixels_per_foot)

    return dist_carried



#arccos of 2 points with a 3rd as a reference to get angle
def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    angle = np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0))
    return int(np.degrees(angle))




#main function for assessing the ejms process
def overall_assess_ejms(ejms_params):
    neck,shoulder_angle,trunk_angle,wrist_angle,trunk_tort,leg_angle,ld,dist_carried,slp,start_time = ejms_params
    
    neck_score = 0 
    if neck>5:
        neck_score+=1
        if neck > 20:
            neck_score +=1
    
    trunk_score = 0
    if trunk_angle>10 or trunk_tort >10:
        trunk_score+=1
        if trunk_angle>20 or trunk_tort>20:
            trunk_score+=1
    
    shoulder_score = 0 
    if shoulder_angle >30:
        shoulder_score+=1
        if shoulder_angle>90:
            shoulder_score+=1
    
    leg_score = 0 
    if leg_angle > 90:
        leg_score+=1
        if leg_angle >20:
            leg_score+=1
    
    slp_score = 0
    if slp > 1:
        slp_score+=1
        if slp>9:
            slp_score+=1
            if slp>11:
                slp_score+=1
                if slp>13:
                    slp_score+=1

    dist_score = 0
    if dist_carried >= 1:
        dist_score+=1
        if dist_carried >=6:
            dist_score+=1 
            if dist_carried >=14:
                dist_score+=1
                if dist_carried >= 21:
                    dist_score+=1
    
    ld_score = 0
    if ld>0:
        ld_score+=1
        if ld>=15:
            ld_score+=1
            if ld>23:
                ld_score+=1
                if ld>36:
                    ld_score+=1
    
    overall = neck_score+trunk_score+shoulder_score+leg_score+slp_score+dist_score+ld_score
    overall_df = pd.DataFrame.from_dict({int(time.time()-start_time):[neck_score,trunk_score,shoulder_score,leg_score,slp_score,dist_score,ld_score,overall]},
    orient='index',columns = ['neck_score','trunk_score','shoulder_score','leg_score','slp_score','dist_score','ld_score','overall'])
    # overall_df.index.name = 'Time(S)'
    return overall_df


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
    dfs,ejms_dfs,left_ejms_dfs,right_ejms_dfs,video_path = datasets
    dfs.index.name = 'Time(S)'

    left_ejms_counts = individual_ejms_counts(left_ejms_dfs)
    right_ejms_counts = individual_ejms_counts(right_ejms_dfs)

    #this needs to be one excel file
    overall_ejms_counts = ejms_counter(ejms_dfs)
    dfs.to_csv(f'{video_path}_report.csv')

    with pd.ExcelWriter(f'{video_path}_EJMS_breakdown.xlsx',mode='w') as writer:  
        left_ejms_dfs.to_excel(writer,sheet_name='left_ejms')
        right_ejms_dfs.to_excel(writer,sheet_name='right_ejms')
        left_ejms_counts.to_excel(writer,sheet_name='left_ejms_counts')
        right_ejms_counts.to_excel(writer,sheet_name='right_ejms_counts')


#this will get the ejms counts of each category
def individual_ejms_counts(df):
    overall_df = df.apply(pd.Series.value_counts).fillna(0).astype(int)
    return overall_df



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

    trunk_tor = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_HIP.value
    ]
    return trunk_tor,left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm


#this defines the distance deltas for vertical and horizontal lifts
def define_distances(mp_pose,landmarks):
    ld = [mp_pose.PoseLandmark.LEFT_WRIST.value,mp_pose.PoseLandmark.LEFT_ANKLE.value]
    top_bottom = [mp_pose.PoseLandmark.LEFT_EAR.value,mp_pose.PoseLandmark.LEFT_ANKLE.value]
    dist = [mp_pose.PoseLandmark.LEFT_ANKLE.value]
    mid_pt = [mp_pose.PoseLandmark.LEFT_HIP.value]
    new_dist =  np.array([landmarks[dist[0]].x,landmarks[dist[0]].y])
    return ld,top_bottom,dist,mid_pt,new_dist



#Calculates the joint angles
def make_angles(landmarks,limb_sets):
    trunk_tor,left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm = limb_sets
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
    
    trunk_tort = fix_angle(calculate_angle(
        (landmarks[trunk_tor[0]].x, landmarks[trunk_tor[0]].y),
        (landmarks[trunk_tor[1]].x, landmarks[trunk_tor[1]].y),
        (landmarks[trunk_tor[2]].x, landmarks[trunk_tor[2]].y)
    ),90)

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
    return left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle,trunk_angle,trunk_tort,neck_angle
