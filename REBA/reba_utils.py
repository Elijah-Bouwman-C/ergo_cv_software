import tkinter as tk
from tkinter import simpledialog,filedialog,messagebox,ttk
import cv2
import numpy as np
import pandas as pd
import time


def calculate_reba_score(angles):
    trunk_tort,neck_angle,upper_arm_angle,lower_arm_angle,trunk_angle,wrist_angle,leg_angle,weight,coupling = angles
    # print(angles)
    score = pd.DataFrame()
    if coupling == 'good':
        score['coupling'] = [0]
    elif coupling == 'fair':
        score['coupling'] = [1]
    elif coupling == 'poor':
        score['coupling'] = [2]
    elif coupling == 'unacceptable':
        score['coupling'] = [3]
    if 20 > neck_angle:
        score['neck'] = [1]
    # Score calculation based on angles
    elif neck_angle >= 20:
        score['neck'] = [2]
    
    if 20 > trunk_angle :
        score['trunk'] = [2]
    elif 60 > trunk_angle > 20:
        score['trunk'] = [3]
    elif trunk_angle > 60:
        score['trunk'] = [4]
    
    if trunk_tort > 15 and 20 > trunk_angle:
        score['trunk'] = [score['trunk'].to_list()[0] + 1]

    if 60 > leg_angle :
        score['leg'] = [1]
    elif leg_angle >= 60:
        score['leg'] = [2]
    
    if 20 > upper_arm_angle :
        score['upper_arm'] = [1]
    elif 45 > upper_arm_angle >= 20:
        score['upper_arm'] = [2]
    elif 90 > upper_arm_angle >= 45:
        score['upper_arm'] = [3]
    elif  upper_arm_angle >= 90:
        score['upper_arm'] = [4]

    if 100 > lower_arm_angle >= 60:
        score['lower_arm'] = [1]
    elif lower_arm_angle <= 60:
        score['lower_arm'] = [2]
    elif lower_arm_angle >= 100:
        score['lower_arm'] = [2]

    if 15 > wrist_angle:
        score['wrist'] = [1]
    elif wrist_angle >= 15:
        score['wrist'] = [2]
    
    if 11 > weight :
        score['force'] = [0]
    elif 22 >= weight >=  11:
        score['force'] = [1]
    elif weight > 22:
        score['force'] = [2]
    score = reba_tables(score)
    return score



def reba_tables(score):
    neck_score = score['neck'].to_list()[0]
    trunk_score = score['trunk'].to_list()[0] 
    leg_score = score['leg'].to_list()[0]
    
    reba_table_a = np.array([
                [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
                [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
                [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
            ])

    score['table_a'] = ([reba_table_a[neck_score-1][trunk_score-1][leg_score-1]])[0] + score['force'].to_list()[0]

    reba_table_b = np.array([
                [[1, 2, 2], [1, 2, 3]],
                [[1, 2, 3], [2, 3, 4]],
                [[3, 4, 5], [4, 5, 5]],
                [[4, 5, 5], [5, 6, 7]],
                [[6, 7, 8], [7, 8, 8]],
                [[7, 8, 8], [8, 9, 9]],
            ])
    ua_score = score['upper_arm'].to_list()[0]
    la_score = score['lower_arm'].to_list()[0]
    wrist_score = score['wrist'].to_list()[0]
    score['table_b'] = ([reba_table_b[ua_score-1][la_score-1][wrist_score-1]])[0] + score['coupling'].to_list()[0] 
    
    reba_table_c = np.array([
                [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
                [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
                [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
                [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
                [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
                [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
                [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
                [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
                [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
                [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
                [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
                [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
            ])
    score['table_c'] = reba_table_c[score['table_b'].to_list()[0]-1][score['table_a'].to_list()[0]-1]
    return score

#reba_table_a[0] = neck
#reba_table_a[0][0] = trunk
#reba_table_a[0][0][0] = legs

#reba_table_b[0] = upper arm
#reba_table[0][0] = lower arm
#reba_table[0][0][0] = wrist




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




#This mostly just filters information input and output 
def get_greatest(left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle,left_wrist,right_wrist):
    dataset = pd.DataFrame.from_dict({'legs':[left_leg_angle,right_leg_angle],'upper_arms':[left_upper_arm_angle,right_upper_arm_angle],'lower_arms':[left_lower_arm_angle,right_lower_arm_angle],'wrists':[left_wrist,right_wrist]})
    maxes = []
    for key in dataset.keys():
        maxes.append(dataset[key].max())
    return maxes



#Assume most joints are set at ~180 degrees
def fix_angle(angle,change=180):
    return abs(angle-change)


#This just saves data neatly
def post_process(datasets,video_path):
    dfs,reba_dfs,both_dfs, = datasets
    dfs.index.name = 'Time(S)'
    reba_dfs.index.name = 'Time(S)'

    #this needs to be one excel file
    
    with pd.ExcelWriter(f'{video_path}_EJMS_breakdown.xlsx',mode='w') as writer:  
        dfs.to_excel(writer,sheet_name='REBA_angles')
        reba_dfs.to_excel(writer,sheet_name='REBA_scores')
        both_dfs.to_excel(writer,sheet_name='Both_sides')
        


#this will get the ejms counts of each category
def individual_ejms_counts(df):
    overall_df = df.apply(pd.Series.value_counts).fillna(0).astype(int)
    return overall_df




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
        mp_pose.PoseLandmark.RIGHT_ANKLE.value
    ]
    right_wrist = [
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_INDEX.value
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
    left_wrist = [
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_INDEX.value
    ]
    return trunk_tor,left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm,right_wrist,left_wrist


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
    trunk_tor,left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm,right_wrist,left_wrist = limb_sets
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
    right_wrist_angle = fix_angle(calculate_angle(
    (landmarks[right_wrist[0]].x, landmarks[right_wrist[0]].y),
    (landmarks[right_wrist[1]].x, landmarks[right_wrist[1]].y),
    (landmarks[right_wrist[2]].x, landmarks[right_wrist[2]].y)
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
    left_wrist_angle = fix_angle(calculate_angle(
        (landmarks[left_wrist[0]].x, landmarks[left_wrist[0]].y),
        (landmarks[left_wrist[1]].x, landmarks[left_wrist[1]].y),
        (landmarks[left_wrist[2]].x, landmarks[left_wrist[2]].y)
    ))
    
    return left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle,trunk_angle,trunk_tort,neck_angle,left_wrist_angle,right_wrist_angle
