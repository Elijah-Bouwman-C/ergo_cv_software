# Lifting Ergo Software 
- This package is mostly meant for lifting/pushing/pulling, although it can be repurposed for the CUSA since much of the code is the same
- app_utils.py contains most of the functions to improve readability
- ergo_ui.py contains the GUI written in pyqt5, which is effectively C++
# CUSA Ergo Software
- Almost a carbon copy of the lifting package, but separated for time
- Mostly checks for 90 or 0 degree angles
-  hand_ergo_ui.py and desk_ui.py are carbon copies of each other
# Hand Ergo Software
- Meant to measure fine movements like soldering, wire stripping, etc.
- hand_app_utils contains the helper functions and will likely need updated to better measure wrist and arm twisting
# Troubleshooting
Since the firewall will most likely block any pip install attempts, download the pre-built wheels from these sources (you may have to search for a wheel built for your python version as these are for windows python 3.11):
- download anaconda distribution from https://www.anaconda.com/download make sure to do a user installation so you don't need admin
- download opencv from here https://files.pythonhosted.org/packages/ec/6c/fab8113424af5049f85717e8e527ca3773299a3c6b02506e66436e19874f/opencv_python-4.10.0.84-cp37-abi3-win_amd64.whl
- download protobuf from here https://files.pythonhosted.org/packages/e1/94/d77bd282d3d53155147166c2bbd156f540009b0d7be24330f76286668b90/protobuf-5.27.3-py3-none-any.whl
- download mediapipe from here https://files.pythonhosted.org/packages/c1/0f/4dc0802131756a9fe4d46d2824352014b85a75baca386cb9e43057f39f15/mediapipe-0.10.14-cp311-cp311-win_amd64.whl

- DO NOT USE SPYDER AS WHEELS ARE NOT COMPATIBLE WITH IT, use VSCode instead from the Software Center
- Afterwards use `pip install --no-deps <file_path>` in the terminal 
- If this doesn't work, you might need to install cmake, which comes with MATLAB 

