from setuptools import setup 
  
setup( 
    name='EJMS-T', 
    version='ErgoVantage', 
    author='Will Anderson', 
    description='A package for human pose estimation specifically for Collins/RTX EHS standards',
    author_email='con.anderson.william@gmail.com',
    install_requires=[ 
        'numpy', 
        'pandas',
        'opencv-python',
        'mediapipe',
        'pyqt5'
        'pyinstaller'
    ], 
) 
