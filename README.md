# Ergonomics Computer Vision Software 

- This software package is based on opencv-python and mediapipe from Google and is meant to replicate the measurements made by a goniometer
- It takes in a .mp4 or .mov file and then maps a skeleton onto their limbs
- Datapoints are extracted and converted into .csv files along with an anonymized recording of the simulation
- These csv files can be analyzed in EHS Smart or sent on to an ergonomist
- EJMS scores can be checked using the REBA software package
  
# Current Goal:
- Repackage data for EHS Smart

# Future Goals:
- Move app to the citrix platform
- Implement Snowflake API to directly input data
- Improve general user interface
- Develop a gesture detection model
- Fix video window sizing issues
