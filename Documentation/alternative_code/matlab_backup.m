clear

% Initialize the PoseEstimator
detector = posenet.PoseEstimator;
height = 5.5; %FEET

% Initialize the webcam
cam = webcam(2);

% Create a video player
player = vision.DeployableVideoPlayer;

% Initialize an empty image
empty_Im = zeros(256, 192, 3, 'uint8');
var_names = {'Time(s)','l_lower_arm','r_lower_arm','l_knee_angle','r_knee_angle','neck','trunk','trunk_tort','lift_dist','carry_dist','slp'};
% Display the empty image
player(empty_Im);
carry_dist = 0;
frame_num = 0;
full_mat = zeros(1,length(var_names));
while player.isOpen

    frame_num = frame_num + 1;
    % Read an image from the webcam
    I = snapshot(cam);
    % Crop the image to fit the network input size of 256x192
    Iinresize = imresize(I, [256 nan]);
    Itmp = Iinresize(:, (size(Iinresize, 2) - 192) / 2 : (size(Iinresize, 2) - 192) / 2 + 192 - 1, :);
    Icrop = Itmp(1:256, 1:192, 1:3);
    
    % Predict pose estimation
    heatmaps = detector.predict(Icrop);
    keypoints = detector.heatmaps2Keypoints(heatmaps);
    %imshow(Icrop);
    
    nose = keypoints(1, :);
    %l
    l_ear = keypoints(17, :);
    l_shoulder = keypoints(6, :);
    l_elbow = keypoints(7, :);
    l_wrist = keypoints(8, :);
    
    l_hip = double(keypoints(12, :));
    l_knee = keypoints(13, :);
    l_ankle = keypoints(14, :);
    
    %r 
    r_ear = keypoints(16, :);
    r_shoulder = double(keypoints(3, :));
    r_elbow = keypoints(4, :);
    r_wrist = keypoints(5, :);
    
    r_hip = double(keypoints(8, :));
    r_knee = keypoints(10, :);
    r_ankle = keypoints(11, :);
    
    %angles
    l_lower_arm = vecangle360(l_shoulder,l_elbow,l_wrist);
    r_lower_arm = vecangle360(r_shoulder,r_elbow,r_wrist);
    
    r_upper_arm = vecangle360(r_hip,r_shoulder,r_elbow);
    l_upper_arm = vecangle360(l_hip,l_shoulder,l_elbow);
    
    r_knee_angle = vecangle360(r_hip,r_knee,r_ankle);
    l_knee_angle = vecangle360(l_hip,l_knee,l_ankle);
    
    neck = abs(max(vecangle360(l_ear,l_shoulder,l_hip),vecangle360(r_ear,r_shoulder,r_hip))-30);
    trunk = max(vecangle360(l_shoulder,l_hip,l_knee),vecangle360(r_shoulder,r_hip,r_knee));
    trunk_tort = abs(vecangle360(r_shoulder,r_hip,l_hip)-90);
    
    pix_per_foot = abs(nose(2)-r_ankle(2))/height;
    lift_dist = max([norm(abs(r_hip-r_wrist)),norm(abs(l_hip-l_wrist))]);
    slp = max([abs(r_hip(1)-r_wrist(1)),abs(l_hip(1)-l_wrist(1))]);
    carry_dist = carry_dist + norm(abs(l_ankle-r_ankle))/pix_per_foot;
    
    % Visualize key points
    
    Iout = detector.visualizeKeyPoints(empty_Im, keypoints);
    imshow(Iout);
    temp_mat = [(frame_num/30),l_lower_arm,r_lower_arm,l_knee_angle,r_knee_angle,neck,trunk,trunk_tort,lift_dist,carry_dist,slp];
    
    if rem(frame_num,30) == 0 
        full_mat = [full_mat;temp_mat];
    end
    
    if ~isOpen(player)  || strcmp(get(gcf, 'CurrentKey'), 'escape')
        break
    end
    
end

% Clean up
clear cam
release(player)
report_mat = array2table(full_mat,'VariableNames',var_names);
writetable(report_mat,'EJMS Report.csv');
close all;

function a = vecangle360(v1,v2,n)
    x = cross(v1,v2);
    c = sign(dot(x,n)) * norm(x);
    CosTheta = atan2d(c,dot(v1,v2));
    a = abs(real(acosd(CosTheta))-180);
end
