from setuptools import setup 
  
setup( 
    name='Collins Aerospace Ergonomics Computer Vision', 
    version='0.1', 
    author='Collins EH&S CR', 
    install_requires=[ 
        'numpy', 
        'pandas',
        'opencv-python',
        'mediapipe',
        'pyqt5'
        'pyinstaller'
    ], 
) 
