import subprocess
# import do_kilosort as dk
import os
import shared.config as config
import numpy as np
import sys
import tkinter as tk
from phy.apps.template import template_gui

bonsaiPath = "C:/Users/miche/AppData/Local/Bonsai/Bonsai.exe"

def collect_data():
    filePath = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Code/online/david_bonsai/workflows/Ephy Full Rig Matrix Merge Test.bonsai"
    subprocess.run([bonsaiPath, filePath])

def kilosort(dataLoc):
    print("kilosort not implemented on the laptop: requires CUDA compatibility")
    pass
    # if dataLoc == None or dataLoc == "":
    #     dataLoc = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_01m00s/"
    # dk.do_kilosort(dataLoc, 385, 'neuropixPhase3B1_kilosortChanMap.mat')
    
def visualise_kilosort(dataLoc):
    if dataLoc == None or dataLoc == "":
        dataLoc = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_05m00s/"
    os.chdir(dataLoc)
    template_gui("kilosort4/params.py")

def open_bonsai():
    filePath = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Code/online/michelle_bonsai/spike_control_loop_dummy.bonsai"
    subprocess.run([bonsaiPath, filePath])

root = tk.Tk()
root.title("Spike Sort GUI")
frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

for i in range(4):
    section = tk.Frame(frame)
    section.grid(row=0, column=i, padx=10)
    match i:
        case 0: 
            button=tk.Button(section, text="Collect Data", command=lambda: collect_data())
        case 1: 
            text_var = tk.StringVar()
            text_var.set("Binary File Location:")
            label = tk.Label(section, textvariable=text_var)
            entry = tk.Entry(section, width=50)
            label.pack(pady=5)
            entry.pack(pady=5)
            button=tk.Button(section, text="Run Kilosort", command=lambda: kilosort(entry.get()))
        case 2: 
            text_var = tk.StringVar()
            text_var.set("Data Outputs Location")
            label = tk.Label(section, textvariable=text_var)
            entry = tk.Entry(section, width=50)
            label.pack(pady=5)
            entry.pack(pady=5)
            button=tk.Button(section, text="Open Output", command=lambda: visualise_kilosort(entry.get()))
        case 3: 
            button=tk.Button(section, text="Run Realtime", command=lambda: open_bonsai())
    button.pack()

root.mainloop()

if __name__ == "__main__":
    root.mainloop()

