import glob
import os
import sys
import tkinter as tk 

def get_wheel_folder    ():
    root = tk.Tk()
    root.withdraw()
    wheel_path = filedialog.askdirectory(title = 'Select download folder')
    wheel_paths = glob(wheel_path, r'\*.whl')
    return wheel_paths

if __name__ == '__main__':
    whl_paths = get_wheel_folder()
    for path in whl_paths:
        os.system(f'pip install --no-deps {path}')
