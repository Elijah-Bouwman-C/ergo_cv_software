import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pandas as pd 
from app_utils import *
from desk_ui import *
import os



def main():
    #Config net model
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    #Config video
    video_path = get_height_weight_video_path() #Custom function in app_utils for GUI
    cap = cv2.VideoCapture(video_path)

    in_height = 5.5
    start_time = time.time()
    last_saved_time = start_time
    current_time = start_time
    previous_time = start_time
    no_flag = True
    dfs = pd.DataFrame(columns = ['neck_angle','upper_arm_angle','lower_arm_angle','trunk_angle',
    'knee_angle'])
    window_name = 'Desk Assessment'
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
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
                #calculate limbs
                trunk_tor,left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm = define_angles(mp_pose) #Custom function in app_utils
                ld,top_bottom,dist,mid_pt,new_dist = define_distances(mp_pose,landmarks) #Custom function in app_utils
                limb_sets = [trunk_tor,left_leg,left_lower_arm,left_upper_arm,trunk,neck,right_leg,right_lower_arm,right_upper_arm]
                left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle,trunk_angle,trunk_tort,neck_angle = make_angles(landmarks,limb_sets) #Custom function in app_utils
                knee_angle,upper_arm_angle,lower_arm_angle = get_greatest(left_leg_angle,right_leg_angle,left_upper_arm_angle,right_upper_arm_angle,left_lower_arm_angle,right_lower_arm_angle) #Custom function in app_utils
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                #save every second
                if current_time - last_saved_time >= 1:
                    data = pd.DataFrame.from_dict({(int(current_time-start_time)):
                    [neck_angle,upper_arm_angle,lower_arm_angle,trunk_angle,knee_angle]}, 
                    orient='index',
                    columns = ['neck_angle','upper_arm_angle','lower_arm_angle','trunk_angle','knee_angle'])
                    data['problematic_areas'] = get_desk_params(data) #Custom function in app_utils
                    dfs = pd.concat((dfs,data),ignore_index=False)
                    print(dfs)
                    last_saved_time = current_time

            cv2.resizeWindow(window_name, 400, 700) 

            cv2.imshow(window_name,frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    dfs.index.name = 'Time(S)'
    dfs.to_csv('chair_assessment.csv')
    cap.release()

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
