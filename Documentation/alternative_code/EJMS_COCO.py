import cv2 as cv
import numpy as np
import argparse
import time 
import pandas as pd



def calculate_pix_height(neck,ankle,person_h):
    if neck and ankle:
        pixel_height = abs(neck[1] - ankle[1])
        pixels_per_foot = pixel_height/person_h
        return pixels_per_foot 




def horizontal_dist(points,prev_points,pixels_per_foot):
    if points!= None:
        pixel_distance = np.linalg.norm(np.array(points)-np.array(prev_points))
        dist_carried = pixel_distance/pixels_per_foot
        return abs(dist_carried)
    return 0



def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    angle = np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0))
    if np.isnan(np.degrees(angle)) != True:
        return int(np.degrees(angle))
    else:
        return 0



def assess_ejms(ejms_params):
    neck,shoulder_angle,trunk,trunk_tort,hip_angle,velocity,ld,dist_carried,slp = ejms_params
    score = 0
    if neck>20:
        score+=1
    if trunk>60 or trunk_tort >45:
        score+=1
    if shoulder_angle >30:
        score+=1
        if shoulder_angle>60:
            score+=1
    if hip_angle > 90:
        score+=1
    if slp > 1:
        score+=1
        if slp>9:
            score+=1
            if slp>11:
                score+=1
                if slp>13:
                    score+=1
    if dist_carried >= 1:
        score+=1
        if dist_carried >=6:
            score+=1 
            if dist_carried >=14:
                score+=1
                if dist_carried >= 21:
                    score+=1
    if ld>0:
        score+=1
        if ld>=15:
            score+=1
            if ld>23:
                score+=1
                if ld>36:
                    score+=1
    return score




def fix_angle(angle,change=180):
    return abs(angle-change)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--params', default = fr'C:\Users\E40078317\Downloads\graph_opt.pb',type=str,help = 'Path to parameters file')
    args = parser.parse_args()
    person_h = 5 + (1/3)
    body_parts = { 'Nose': 0, 'Neck': 1, 'RShoulder': 2, 'RElbow': 3, 'RWrist': 4,
                'LShoulder': 5, 'LElbow': 6, 'LWrist': 7, 'RHip': 8, 'RKnee': 9,
                'RAnkle': 10, 'LHip': 11, 'LKnee': 12, 'LAnkle': 13, 'REye': 14,
                'LEye': 15, 'REar': 16, 'LEar': 17, 'Background': 18 }

    pose_pairs = [ ['Neck', 'RShoulder'], ['Neck', 'LShoulder'], ['RShoulder', 'RElbow'],
                ['RElbow', 'RWrist'], ['LShoulder', 'LElbow'], ['LElbow', 'LWrist'],
                ['Neck', 'RHip'], ['RHip', 'RKnee'], ['RKnee', 'RAnkle'], ['Neck', 'LHip'],
                ['LHip', 'LKnee'], ['LKnee', 'LAnkle'], ['Neck', 'Nose'], ['Nose', 'REye'],
                ['REye', 'REar'], ['Nose', 'LEye'], ['LEye', 'LEar'] ]

    in_width = args.width
    in_height = args.height

    net = cv.dnn.readNetFromTensorflow(args.params)

    cap = cv.VideoCapture(args.input if args.input else 0)


    neck_angle,trunk_angle, trunk_tort,l_hip_angle, r_hip_angle, l_elbow_angle, r_elbow_angle, l_shoulder_angle,r_shoulder_angle,velocity,ld,dist_carried,reba_score,slp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    dfs = pd.DataFrame(columns = ['neck_angle','trunk_angle','l_hip_angle','r_hip_angle','l_elbow_angle','r_elbow_angle',
            'l_shoulder_angle','r_shoulder_angle','velocity','LD','horizontal_dist','REBA'])
    start_time = time.time()
    last_saved_time = start_time

    previous_neck_position,prev_points = [None,None]
    previous_time = start_time

    while cv.waitKey(1) < 0:
        current_time = time.time()
        has_frame, frame = cap.read()

        if not has_frame:
            cv.waitKey()
            break
        
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (in_width, in_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(body_parts) == out.shape[1])

        points = []
        for i in range(len(body_parts)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frame_width * point[0]) / out.shape[3]
            y = (frame_height * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > args.thr else None)
        
        for pair in pose_pairs:
            part_from = pair[0]
            part_to = pair[1]
            assert(part_from in body_parts)
            assert(part_to in body_parts)

            idFrom = body_parts[part_from]
            idTo = body_parts[part_to]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        if points:
            #Upper right arm
            if  points[body_parts['RHip']] and points[body_parts['RShoulder']] and points[body_parts['RElbow']]:
                r_shoulder_angle = calculate_angle(points[body_parts['RHip']], points[body_parts['RShoulder']], 
                                                    points[body_parts['RElbow']])

            #Upper left arm
            if  points[body_parts['LHip']] and points[body_parts['LShoulder']] and points[body_parts['LElbow']]:
                l_shoulder_angle = calculate_angle(points[body_parts['LHip']], points[body_parts['LShoulder']], 
                                                points[body_parts['LElbow']])
                                                
            #Lower right arm
            if  points[body_parts['RShoulder']] and points[body_parts['RElbow']] and points[body_parts['RWrist']]:
                r_elbow_angle = fix_angle(
                    calculate_angle(points[body_parts['RShoulder']], points[body_parts['RElbow']], points[body_parts['RWrist']]))
                
            #Lower left arm
            if  points[body_parts['LShoulder']] and points[body_parts['LElbow']] and points[body_parts['LWrist']]:
                l_elbow_angle = fix_angle(
                    calculate_angle(points[body_parts['LShoulder']], points[body_parts['LElbow']], points[body_parts['LWrist']]))

            #Left leg
            if  points[body_parts['RHip']] and points[body_parts['RKnee']] and points[body_parts['RAnkle']]:
                r_hip_angle = fix_angle(
                    calculate_angle(points[body_parts['RHip']], points[body_parts['RKnee']], points[body_parts['RAnkle']]))
                
            #Right Leg
            if  points[body_parts['LHip']] and points[body_parts['LKnee']] and points[body_parts['LAnkle']]:
                l_hip_angle = fix_angle(
                    calculate_angle(points[body_parts['LHip']], points[body_parts['LKnee']], points[body_parts['LAnkle']]))
                
            #Trunk 
            if  points[body_parts['Neck']] and points[body_parts['RHip']] and points[body_parts['RKnee']]:
                trunk_angle = fix_angle(
                    calculate_angle(points[body_parts['Neck']], points[body_parts['RHip']], points[body_parts['RKnee']]))
                
            #Trunk torsion
            if points[body_parts['REar']] and points[body_parts['RShoulder']] and points[body_parts['RHip']]:
                trunk_tort = fix_angle(
                    calculate_angle(points[body_parts['REar']], points[body_parts['RShoulder']], points[body_parts['RHip']])+25)

            #Neck
            if  points[body_parts['RHip']] and points[body_parts['RShoulder']] and points[body_parts['REar']]:
                neck_angle = fix_angle(
                    calculate_angle(points[body_parts['RHip']], points[body_parts['Neck']], points[body_parts['REar']]))
                
            #Vertical distance 
            if  points[body_parts['RAnkle']] != None and points[body_parts['Neck']] != None and points[body_parts['RWrist']] != None:
                pixels_per_foot = calculate_pix_height(points[body_parts['Neck']],points[body_parts['RAnkle']],person_h)
                ld = abs(points[body_parts['RWrist']][1] - points[body_parts['RAnkle']][1])/pixels_per_foot
            
            #Horizontal distance
                if prev_points == None:
                    prev_points = points[body_parts['Neck']]
                dist_carried += horizontal_dist(points[body_parts['Neck']],prev_points,pixels_per_foot)
                slp = horizontal_dist(points[body_parts['RHip']],points[body_parts['RWrist']],pixels_per_foot)
            #Velocity 
            if points[body_parts['Neck']]:
                neck_position = points[body_parts['Neck']]
                if previous_neck_position == None:
                    previous_neck_position=neck_position
                time_difference = current_time - previous_time
                velocity = np.linalg.norm(np.array(neck_position)-np.array(previous_neck_position))/time_difference
                previous_neck_position = neck_position
                previous_time = current_time
            
            ejms = [neck_angle,r_shoulder_angle,trunk_angle,trunk_tort,r_hip_angle,velocity,ld,dist_carried,slp]
            ejms = assess_ejms(ejms)
            cv.putText(frame, f'EJMS Score: {ejms}', (10, 220), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            t, _ = net.getPerfProfile()
            #Save info every 1 seconds 
            if current_time - last_saved_time >= 1:
                data = pd.DataFrame.from_dict({(current_time-start_time):
                [neck_angle,r_shoulder_angle,trunk_angle,trunk_tort,r_hip_angle,velocity,ld,dist_carried,slp,ejms]}, 
                orient='index',
                columns = ['neck_angle','r_shoulder_angle','trunk_angle','trunk_tort','r_hip_angle','velocity','ld','dist_carried','slp','EJMS Score'])
                
                
                dfs = pd.concat((dfs,data),ignore_index=False)
                last_saved_time = current_time

        # if current_time - start_time >= 30:
        #     break


        cv.imshow('REBA Score with OpenCV', frame)

    dfs.to_csv('dataset_ergo.csv')
    cv.destroyAllWindows()




if __name__ == '__main__':
    main()
