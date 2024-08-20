clear

% Initialize the PoseEstimator
detector = posenet.PoseEstimator;

% Initialize the webcam
cam = webcam(2);

% Create a video player
player = vision.DeployableVideoPlayer;

% Initialize an empty image
I = zeros(256, 192, 3, 'uint8');

% Display the empty image
player(I);

while player.isOpen
    % Read an image from the webcam
    I = snapshot(cam);

    % Crop the image to fit the network input size of 256x192
    Iinresize = imresize(I, [256 nan]);
    Itmp = Iinresize(:, (size(Iinresize, 2) - 192) / 2 : (size(Iinresize, 2) - 192) / 2 + 192 - 1, :);
    Icrop = Itmp(1:256, 1:192, 1:3);
    
    % Predict pose estimation
    heatmaps = detector.predict(Icrop);
    keypoints = detector.heatmaps2Keypoints(heatmaps);

    % Calculate vectors u and v
    shoulder = double(keypoints(2, :));
    elbow = keypoints(3, :);
    wrist = keypoints(4, :);
    
    
    

    % Compute the dot product
    CosTheta = vecangle360(shoulder,elbow,wrist);
    
    % Calculate the angle in degrees
    ThetaInDegrees = real(acosd(CosTheta));

    disp(['Angle between forearm segments: ', num2str(ThetaInDegrees), ' degrees']);

    % Visualize key points
    
    Iout = detector.visualizeKeyPoints(Icrop, keypoints);
    imshow(Iout);

    if ~isOpen(player)
        break
    end
end

% Clean up
clear cam
release(player)


function a = vecangle360(v1,v2,n)
x = cross(v1,v2);
c = sign(dot(x,n)) * norm(x);
a = atan2d(c,dot(v1,v2));
end


