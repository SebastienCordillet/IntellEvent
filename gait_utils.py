import numpy as np
from scipy.signal import find_peaks
from sklearn import preprocessing
import http.client
import json
import pandas as pd
import os
import re
import sys
import threading
import moveck_bridge_btk as btk
import onnxruntime as rt

# Define a threshold [0-1] when events should be detected based on the prediction probability
# Increase min_peak_threshold if ghost events appear
# "Real" IC/FO events should be > 0.5.

base_frequency = 150

def intellEvent_c3d(file_name, subject_name, marker_list,min_peak_threshold = 0.3):
    x_traj, y_traj, z_traj = [], [], []

    #Load C3D file []
    acq=btk.btkReadAcquisition(file_name)
    start_frame=int(btk.btkGetFirstFrame(acq))
    end_frame=int(btk.btkGetLastFrame(acq))

    #Load traj for progression Axis [] : ancienne version marker = vicon.GetTrajectory(subject_name, "LHEE")
    marker,_=btk.btkGetPoint(acq,marker_list[0])
    # Check for progression Axis
    prog_x = marker[0][start_frame:end_frame - 1]
    prog_y = marker[1][start_frame:end_frame - 1]

    # Get the corresponding index for each marker name in 'marker_list'
    for marker_name in marker_list:
        xyz,_ = btk.btkGetPoint(acq,marker_name)
        x_traj.append(xyz[start_frame:end_frame,0])
        y_traj.append(xyz[start_frame:end_frame,1])
        z_traj.append(xyz[start_frame:end_frame,2])
    print(btk.btkGetPoints(acq))
    # The current best model uses the x and z axis velocity for the IC model
    # and the x, y, and z axis velocity for the FO model
    # first component should be forward axis progression
    prog_x = marker[start_frame:end_frame,0]
    prog_y = marker[start_frame:end_frame,1]

    # The current best model uses the x and z axis velocity for the IC model
    # and the x, y, and z axis velocity for the FO model
    # first component should be forward axis progression
    if np.mean(np.abs(prog_x)) > np.mean(np.abs(prog_y)):
        ic_traj = np.concatenate([x_traj, z_traj])
        fo_traj = np.concatenate([x_traj, y_traj, z_traj])
    else:
        ic_traj = np.concatenate([y_traj, z_traj])
        fo_traj = np.concatenate([y_traj, x_traj, z_traj])

    # x and y-coordinates need to be standardized depending on the starting direction,
    # z coordinates are always the same
    if any(ic_traj[0, 0:10] < 0) or any(ic_traj[3, 0:10] < 0):
        ic_traj[0:6, :] = (ic_traj[0:6, :] - np.mean(ic_traj[0:6, :], axis=1).reshape(6,1)) * (-1)
        fo_traj[0:12, :] = (fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)) * (-1)

    # calculate the first derivative (= velocity) of the trajectories
    ic_velo = np.gradient(ic_traj, axis=1)
    fo_velo = np.gradient(fo_traj, axis=1)

    # standardize between 0.1 and 1.1 for the machine learning algorithm (zeros will be ignored!)
    ic_velo = preprocessing.minmax_scale(ic_velo, feature_range=(0.1, 1.1), axis=1)
    fo_velo = preprocessing.minmax_scale(fo_velo, feature_range=(0.1, 1.1), axis=1)


    #Down / up sampling?
    cam_frequency = btk.btkGetPointFrequency(acq)
    if cam_frequency != base_frequency:
        rs_ic_velo = resample_data(ic_velo, cam_frequency, base_frequency).transpose()
        rs_fo_velo = resample_data(fo_velo, cam_frequency, base_frequency).transpose()
    else:
        rs_ic_velo = ic_velo
        rs_fo_velo = fo_velo

    # both 'ic_velo' and 'fo_velo' should be in the shape (num_features, num_frames) (e.g. (12, 500) or (18, 500))
    # for the prediction we need the shape of (num_samples, num_frames, num_features)
    # num_samples = 1, num_frames = length of trial (e.g. 500), num_features = velocity of trajectories (e.g. 12 or 18)
    # check with rs_ic_velo.shape
    rs_ic_velo = reshape_data(rs_ic_velo) #rs_ic_velo
    rs_fo_velo = reshape_data(rs_fo_velo) #rs_fo_velo


    # Multithreading to run both predictions at the same time
    # speeds up processing
    t1 = threading.Thread(target=ic_pred, args=(rs_ic_velo.tolist(), ic_traj[0:6], subject_name, start_frame, acq, cam_frequency,min_peak_threshold))
    t2 = threading.Thread(target=fo_pred, args=(rs_fo_velo.tolist(), ic_traj[0:6], subject_name, start_frame, acq, cam_frequency,min_peak_threshold))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # print(f"Events : '{btk.btkGetEvents(acq)}'")
    newfile_name=re.sub(r".c3d", "_e.c3d",file_name)
    btk.btkWriteAcquisition(acq,newfile_name)
    btk.btkCloseAcquisition(acq)
    

def reshape_data(traj):
    """
    Reshape a 2D numpy array into the required input format for the neural network.

    Parameters:
        traj (numpy.ndarray): A 2D array of shape (num_features, num_frames).

    Returns:
        numpy.ndarray: Reshaped array of shape (num_samples, num_frames, num_features).
    """
    rs_traj = np.transpose(np.array(traj).reshape(1, traj.shape[0], traj.shape[1]), (0, 2, 1))
    return rs_traj


def get_prediction(typeOfEvent, rs_velo):
    """
    Parameters :
        typeOfEvent (str): foot strike ("ic") or foot off (fo)
        rs_velo (numpy.ndarray): Reshaped velocity data
    """
    providers = ['CPUExecutionProvider']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if typeOfEvent=="ic":
        model_path = os.path.join(script_dir, "models", "ic_intellevent.onnx")
    elif typeOfEvent=="fo":
        model_path = os.path.join(script_dir, "models", "fo_intellevent.onnx")
    else:
        raise(Exception("Type of event is not supported only accept foot strike (ic) and foot off (fo)."))
    
    model= rt.InferenceSession(model_path, providers=providers)

    prediction = model.run(['time_distributed'], {"input_1": rs_velo})

    return prediction



def ic_pred(rs_velo, x_traj, subject_name, start_frame, acq, cam_frequency, min_peak_threshold):
    """
    predict IC events in c3d acquisition and add events
    """
    ic_preds=get_prediction("ic",rs_velo)
    set_ic_events(ic_preds[0][0], x_traj[0, :], x_traj[3, :], subject_name, start_frame, acq, cam_frequency, min_peak_threshold)
    return True


def fo_pred(rs_velo, x_traj, subject_name, start_frame, acq, cam_frequency,min_peak_threshold):
    """
    predict FO events in c3d acquisition and add events
    """
    fo_preds=get_prediction("fo",rs_velo)
    set_fo_events(fo_preds[0][0], x_traj[0, :], x_traj[3, :], subject_name, start_frame, acq, cam_frequency,min_peak_threshold)
    return True


def set_ic_events(ic_preds, l_heel, r_heel, subject_name, start_frame, acq, cam_frequency,min_peak_threshold):
    """
    Add IC events in c3d acquisition using the predictions.
    """
    # print(f"IC prediction : '{ic_preds}'")
    loc, _ = find_peaks(ic_preds[:,1], height=min_peak_threshold, distance=25)
    loc = np.ceil((loc / base_frequency) * cam_frequency).astype(int)
    for ic in loc:
        if l_heel[ic] < r_heel[ic]:
            btk.btkAppendEvent(acq,"Foot Strike",(ic + start_frame)/cam_frequency,"Left ",subject_name,"", 1)
            #vicon.CreateAnEvent(subject_name, "Left", "Foot Strike", int(ic + start_frame), 0.0)
        else:
            btk.btkAppendEvent(acq,"Foot Strike",(ic + start_frame)/cam_frequency,"Right ",subject_name,"", 1)
            #vicon.CreateAnEvent(subject_name, "Right", "Foot Strike", int(ic + start_frame), 0.0)


def set_fo_events(fo_preds, l_heel, r_heel, subject_name, start_frame, acq, cam_frequency,min_peak_threshold):
    """
    Add FO events in c3d acquisition using the predictions.
    """
    # print(f"FO prediction : '{fo_preds}'")
    loc, _ = find_peaks(fo_preds[:, 1], height=min_peak_threshold, distance=25)
    loc = np.ceil((loc / base_frequency) * cam_frequency).astype(int)
    for fo in loc:
        if l_heel[fo] > r_heel[fo]:
            btk.btkAppendEvent(acq,"Foot Off",(fo + start_frame)/cam_frequency,"Left ",subject_name,"", 2)
        else:
            btk.btkAppendEvent(acq,"Foot Off",(fo + start_frame)/cam_frequency,"Right ",subject_name,"", 2)


def get_trial_infos(c3d_trial):
    """
    Extract trial information from the C3D trial data.
    """
    cam_frequency=btk.btkGetPointFrequency(c3d_trial)
    return cam_frequency


def resample_data(traj, sample_frequ, frequ_to_sample):
    """
    Resample the data to the desired frequency.
    """
    period = '{}N'.format(int(1e9 / sample_frequ))
    index = pd.date_range(0, periods=len(traj[0, :]), freq=period)
    resampled_data = [pd.DataFrame(val, index=index).resample('{}N'.format(int(1e9 / frequ_to_sample))).mean() for val
                      in traj]
    resampled_data = [np.array(traj.interpolate(method='linear')) for traj in resampled_data]
    resampled_data = np.concatenate(resampled_data, axis=1)
    return resampled_data