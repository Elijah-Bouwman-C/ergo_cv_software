import cv2
import mediapipe as mp
import time 
import pandas as pd
import numpy as np

# gets count of number of wrist flexions
def wrist_counts(dfs):
  freq = 0 
  freq_dfs = dfs[['left_wrist','right_wrist']]
  freq_dfs['condition'] = (freq_dfs > 30).any(axis=1)
  conditions = freq_dfs['condition'].to_list()
  prev_condition = False
  for condition in conditions:
    if prev_condition == False and condition == True:
      freq+=1
    prev_condition = condition
  return freq


#this counts the number of pinches grips from boolean
def pinch_counts(dfs,side):
  freq = 0
  conditions = dfs[f'{side}_pinching'].to_list()
  prev_condition = False #Set to open
  for condition in conditions:
    if prev_condition == False and condition == True: #if was open now closed
      freq+=1
    prev_condition = condition
  return freq

#main function for assessing the ejms process
def ejms_assess(dfs):
  freq = 0
  freq_counts = pd.DataFrame()
  keyset = dfs.keys()
  freq_dfs = dfs.drop(['left_wrist','right_wrist','r_pinching','l_pinching'],axis=1)
  freq_dfs['condition'] = (freq_dfs > 10).any(axis=1)
  conditions = freq_dfs['condition'].to_list()
  prev_condition = False
  for condition in conditions:
    if prev_condition == False and condition == True:
      freq+=1
    prev_condition = condition
  freq_counts['fingers_counts'] = [freq]
  freq_counts['wrists_counts'] = [wrist_counts(dfs)]
  freq_counts['left_pinch_counts'] = pinch_counts(dfs,'l')
  freq_counts['right_pinch_counts'] = pinch_counts(dfs,'r')

  return freq_counts


#Assume most joints are set at ~180 degrees
def fix_angle(angle,change=180):
  return abs(angle-change)


# Checks for distance between thumb and index in pixels
def check_pinch(landmarks,mp_hands):
  thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
  index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
  dist = horizontal_dist(thumb_tip,index_tip)

  if  dist < .05:
    return True
  else:
    return False


#uses the magnitude of 2 points to get distance
def horizontal_dist(points,prev_points):
    points = np.array(points)
    prev_points = np.array(prev_points)
    pixel_distance = np.linalg.norm(np.array(points)-np.array(prev_points))
    dist_carried = abs(pixel_distance)

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


#main function to get finger angles
def calculate_fin_angles(landmark,start_time,hand_side,mp_hands):
  angles = {}
  thumb_angle = fix_angle(calculate_angle(landmark[mp_hands.HandLandmark.THUMB_CMC],
  landmark[mp_hands.HandLandmark.THUMB_MCP],
  landmark[mp_hands.HandLandmark.THUMB_IP]))

  index_angle = fix_angle(calculate_angle(landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
  landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
  landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]))

  middle_angle = fix_angle(calculate_angle(landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
  landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
  landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]))
  
  ring_angle = fix_angle(calculate_angle(landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
  landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
  landmark[mp_hands.HandLandmark.RING_FINGER_DIP]))

  pinky_angle = fix_angle(calculate_angle(landmark[mp_hands.HandLandmark.PINKY_MCP],
  landmark[mp_hands.HandLandmark.PINKY_PIP],
  landmark[mp_hands.HandLandmark.PINKY_DIP]))

  pinching = check_pinch(landmark,mp_hands)
  wrist_p = landmark[mp_hands.HandLandmark.WRIST]

  return thumb_angle,index_angle,middle_angle,ring_angle,pinky_angle,pinching,wrist_p


#main function to get wrist angles
def get_wrist_angle(mp_pose,landmarks,start_time):
  right_wrist = [
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_INDEX.value,   
  ]
  
  right_wrist_angle = fix_angle(calculate_angle(
    (landmarks[right_wrist[0]].x, landmarks[right_wrist[0]].y),
    (landmarks[right_wrist[1]].x, landmarks[right_wrist[1]].y),
    (landmarks[right_wrist[2]].x, landmarks[right_wrist[2]].y)
  ))
  
  left_wrist = [
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_INDEX.value,   
  ]
  
  left_wrist_angle = fix_angle(calculate_angle(
    (landmarks[left_wrist[0]].x, landmarks[left_wrist[0]].y),
    (landmarks[left_wrist[1]].x, landmarks[left_wrist[1]].y),
    (landmarks[left_wrist[2]].x, landmarks[left_wrist[2]].y)
  ))
  return left_wrist_angle,right_wrist_angle



# def pinch_counts(dfs,name):
#   prev = False
#   counts = []
#   countset =  dfs[name].to_list()
#   for count in countset:
#     if count:
#       if count != prev:
#         counts.append(1)
#       else:
#         counts.append(0)
#     else:
#       counts.append(0)
#   print(counts)
#   return counts



