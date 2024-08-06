import cv2
import mediapipe as mp
from hand_app_utils import *
import time 
import pandas as pd
from hand_ergo_ui import *
import os



def main():
    #initialize neural network 
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands 
    mp_pose = mp.solutions.pose
    #initialize variables 
    current_time = time.time()
    start_time = current_time
    last_saved_time = current_time
    wrist_positions = []
    dfs = pd.DataFrame()
    left_flag = False
    right_flag = False

    #initialize videos
    video_path = get_height_weight_video_path() #custom function in hand_app_utils    
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    pose = mp_pose.Pose(model_complexity=0,min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f'{os.path.basename(video_path)}.mp4'
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path,fourcc,15,(vid_width,vid_height))
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        #frame processing stuff
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time()
        
        hand_results = hands.process(frame)
        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            #draw the hands
            for hand_landmarks,handedness in zip(hand_results.multi_hand_landmarks,hand_results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                landmarks = [(lm.x,lm.y,lm.z) for lm in hand_landmarks.landmark]
                hand_side = handedness.classification[0].label
                #calculate finger angles
                if hand_side == 'Left':
                    left_flag = True
                    r_thumb_angle,r_index_angle,r_middle_angle,r_ring_angle,r_pinky_angle,r_pinching,r_wrist_p = calculate_fin_angles(landmarks,start_time,hand_side,mp_hands) #custom function in hands_ergo_utils    
                
                if hand_side == 'Right':
                    right_flag = True
                    l_thumb_angle,l_index_angle,l_middle_angle,l_ring_angle,l_pinky_angle,l_pinching,l_wrist_p = calculate_fin_angles(landmarks,start_time,hand_side,mp_hands) #custom function in hand_app_utils
        #draw the body and calculate wrist angles
        pose_results = pose.process(frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_wrist,right_wrist = get_wrist_angle(mp_pose,landmarks,start_time) #custom function in hand_app_utils    
        #save every second
        if current_time - last_saved_time >= 1:
            if left_flag == True and right_flag == True:
                dfset = pd.DataFrame.from_dict({(int(time.time()-start_time)):[r_thumb_angle,r_index_angle,r_middle_angle,r_ring_angle,r_pinky_angle,r_pinching,l_thumb_angle,l_index_angle,l_middle_angle,l_ring_angle,l_pinky_angle,l_pinching,left_wrist,right_wrist]},
                    orient='index',columns = ['r_thumb_angle','r_index_angle','r_middle_angle','r_ring_angle','r_pinky_angle','r_pinching','l_thumb_angle','l_index_angle','l_middle_angle','l_ring_angle','l_pinky_angle','l_pinching','left_wrist','right_wrist'])
                left_flag = False
                right_flag = False
                
                dfs = pd.concat((dfs,dfset),ignore_index=False)
                last_saved_time = current_time
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        out.write(frame)

        cv2.imshow('MediaPipe Hands', frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    freq_counts = ejms_assess(dfs) #custom function in hand_app_utils    
    with pd.ExcelWriter(f'{video_path}_hand_breakdown.xlsx',mode='w') as writer:  
        dfs.to_excel(writer,sheet_name='hand_ejms')
        freq_counts.to_excel(writer,sheet_name ='frequency_counts')

if __name__ == '__main__':
  main()
