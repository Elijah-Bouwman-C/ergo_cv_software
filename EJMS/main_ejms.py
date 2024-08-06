import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import pandas as pd
from ergo_ui import *
from app_utils import *



def main():
    
    #Config net model
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    # mp_face_detection = mp.solutions.face_detection

    #Config video
    height, weight, video_path = get_height_weight_video_path()
    
    cap = cv2.VideoCapture(video_path)
    #initialize variables
    prev_dist = None
    dist_carried = 0
    start_time = time.time()
    last_saved_time = start_time
    current_time = start_time
    previous_time = start_time
    no_flag = True
    dfs = pd.DataFrame(columns = ['neck_angle','upper_arm_angle','lower_arm_angle','trunk_angle','trunk_tort',
    'leg_angle','lifting distance (ft)','dist_carried(ft)','standard lifting point(ft)','EJMS Score'])
    ejms_dfs = pd.DataFrame(columns =['neck_score','trunk_score','shoulder_score','leg_score','slp_score','dist_score','ld_score','overall'])
    left_ejms_dfs = pd.DataFrame(columns =['neck_score','trunk_score','shoulder_score','leg_score','slp_score','dist_score','ld_score','overall'])
    right_ejms_dfs = pd.DataFrame(columns =['neck_score','trunk_score','shoulder_score','leg_score','slp_score','dist_score','ld_score','overall'])
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=1)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image_rgb)
            
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                trunk_tor,left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm = define_angles(mp_pose)
                ld,top_bottom,dist,mid_pt,new_dist = define_distances(mp_pose,landmarks)
                limb_sets = [trunk_tor,left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm]
                
                if no_flag == True:
                    prev_dist = new_dist
                    no_flag = False
                
                height_pix =  abs(landmarks[top_bottom[0]].y-landmarks[top_bottom[1]].y)/height
                ld = abs(landmarks[ld[0]].y-landmarks[ld[1]].y)/height_pix
                slp = (abs(landmarks[dist[0]].x-landmarks[mid_pt[0]].x)/height_pix)
                left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle,trunk_angle,trunk_tort,neck_angle = make_angles(landmarks,limb_sets)
                leg_angle,upper_arm_angle,lower_arm_angle = get_greatest(left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle)

                # Calculate overall EJMS score
                params = [neck_angle,upper_arm_angle,lower_arm_angle,trunk_angle,trunk_tort,leg_angle,ld,dist_carried,slp,start_time]
                left_params = [neck_angle,left_upper_arm_angle,left_lower_arm_angle,trunk_angle,trunk_tort,left_leg_angle,ld,dist_carried,slp,start_time]
                right_params = [neck_angle,right_upper_arm_angle,right_lower_arm_angle,trunk_angle,trunk_tort,right_leg_angle,ld,dist_carried,slp,start_time]
                
                ejms_df = overall_assess_ejms(params) 
                right_ejms = overall_assess_ejms(right_params)
                left_ejms = overall_assess_ejms(left_params)

                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)

                # Draw the pose annotation on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # save every second
                if current_time - last_saved_time >= 1:

                    #Calculate Distances metric 
                    dist_carried += horizontal_dist(new_dist,prev_dist,height_pix)
                    
                    data = pd.DataFrame.from_dict({(int(current_time-start_time)):
                    [neck_angle,upper_arm_angle,lower_arm_angle,trunk_angle,trunk_tort,leg_angle,round(ld,2),round(dist_carried,2),round(slp,2),ejms_df['overall']]}, 
                    orient='index',
                    columns = ['neck_angle','upper_arm_angle','lower_arm_angle','trunk_angle','trunk_tort',
                    'leg_angle','lifting distance (ft)','dist_carried(ft)','standard lifting point(ft)','EJMS Score'])
                    
                    ejms_dfs = pd.concat((ejms_dfs,ejms_df),ignore_index=False)
                    left_ejms_dfs = pd.concat((left_ejms_dfs,left_ejms),ignore_index=False)
                    right_ejms_dfs = pd.concat((right_ejms_dfs,right_ejms),ignore_index=False)
                    dfs = pd.concat((dfs,data),ignore_index=False)
                    
                    last_saved_time = current_time
            # frame = blur_face(image_rgb,frame,face_detection)
            cv2.imshow('Press "q" to stop video and exit window', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    #final save
    datasets = [dfs,ejms_dfs,left_ejms_dfs,right_ejms_dfs,video_path]
    post_process(datasets)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
