# Alternative Main Functions
- These are both proof of concept programs developed in ~1-2 days with limited capability, but serve as a starting point should there be issues with implementation
# MATLAB Code
- `matlab_backup.m` is identical to `EJMS_main.py` it is only missing a GUI and the EJMS scoring step
- It uses the COCO framework and there is a great tutorial here: https://www.youtube.com/watch?v=qR9065BXSLc
- Might require the toolbox here:
  https://www.mathworks.com/matlabcentral/fileexchange/76860-human-pose-estimation-with-deep-learning
- Might also require this for installation
  https://www.mathworks.com/matlabcentral/answers/389223-how-do-i-silently-install-support-packages-in-matlab-r2018a-or-newer#Downloaded
- Highly recommend updating to the latest MATLAB and then using this framework:
 https://www.mathworks.com/help/vision/ug/multi-object-tracking-and-human-pose-estimation.html
# Python Code 
- You are going to need to download the neural network from here (Although I highly recommend using any model from OpenCV)
  https://github.com/quanhua92/human-pose-estimation-opencv/tree/master
- Here is a repository of different available pose estimation models
  https://github.com/opencv/opencv_extra/tree/4.x/testdata/dnn
