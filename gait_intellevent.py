#from viconnexusapi import ViconNexus
from sklearn import preprocessing
import numpy as np #conda install numpy==1.18.5
from gait_utils import ic_pred, fo_pred, reshape_data, resample_data, intellEvent_c3d
import threading
import sys
import moveck_bridge_btk as btk
import os
import tkinter as tk
from tkinter import filedialog

# marker names which are used for the algorithm
# adapt names to how they are called in Vicon
# marker_list = ["LHEE", "LTOE", "LANK", "RHEE", "RTOE", "RANK"] # VICON PIG
marker_list=["L_FCC","L_FM5","L_FAL","R_FCC","R_FM5","R_FAL"] # QTM IOR
base_frequency = 150



if __name__=='__main__':
    root = tk.Tk()
    root.withdraw()  # Cache la fenêtre principale

    fichiers = filedialog.askopenfilenames(
        title="Sélectionner des fichiers .c3d",
        filetypes=[("Fichiers C3D", "*.c3d")]
    )

    for fichier in fichiers:
        intellEvent_c3d(fichier, "intellEvent",marker_list)

    