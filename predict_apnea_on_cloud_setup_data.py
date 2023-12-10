# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import fnmatch
from os import listdir
from sklearn import preprocessing
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
import keras
from Tests.NN.create_apnea_count_AHI_data import compute_respiration, delays, getSetupRespirationCloudDBDebugWithTimestamp, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebug, getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, compute_respiration, compute_phase
from Tests.NN.create_apnea_count_AHI_data_regression_cloud_data_no_ref import get_empty_seconds_mb, create_AHI_regression_training_data_no_ref
import matplotlib.pyplot as plt
import glob
import scipy.signal as sp

db = DB()
home = '/Neteera/Work/homes/dana.shavit/Research/analysis/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707/acc_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_selected_sessions/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_amp_with_filter7_selected_sessions/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_zc_10sec_ec_win_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_10sec_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_min_time_for_output_10/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_min_time_for_output_10_3/'

base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/apnea_2806/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/prepared_data/ahi_data_tamir/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/prepared_data/ahi_data_stiched/scaled/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/prepared_data/ahi_data_stiched_2611/scaled/'
base_path = '/Neteera/Work/homes/tamir.golan/python_code/Apnea/pythondev/Data/scaled/'

fn = 'ahi' + '.png'
posture_class = {'Supine':0, 'Left':2, 'Right':3, 'Prone':1}
color_radar = {1:'red',2:'green', 3:'blue', 4:'magenta'}
setups = 'all'


def getApneaSegments(setup:int, respiration: np.ndarray, fs_new: float):
    """compute apnea segments per session"""
    apnea_ref = None
    db.update_mysql_db(setup)
    if db.mysql_db  == 'neteera_cloud_mirror':
        apnea_ref = load_reference(setup, 'apnea', db)
    else:
        if setup in db.setup_nwh_benchmark():
            apnea_ref = load_reference(setup, 'apnea', db)
        else:
            session = db.session_from_setup(setup)
            setups = db.setups_by_session(session)
            for s in setups:
                if s in db.setup_nwh_benchmark():
                    apnea_ref = load_reference(s, 'apnea', db)

    #print(sess, type(apnea_ref))
    if apnea_ref is None:
        return None

    if isinstance(apnea_ref, pd.core.series.Series):
        apnea_ref = apnea_ref.to_numpy()
    apnea_from_displacement = np.zeros(len(apnea_ref) * fs_new)

    apnea_segments = []

    for i in range(len(apnea_ref) - 1):
        if apnea_ref[i] not in apnea_class.keys():
            apnea_from_displacement[i * fs_new:(i + 1) * fs_new] = -1
        else:
            apnea_from_displacement[i * fs_new:(i + 1) * fs_new] = apnea_class[apnea_ref[i]]

    if apnea_from_displacement[0] == -1:
        apnea_diff = np.diff(apnea_from_displacement, prepend=-1)
    else:
        apnea_diff = np.diff(apnea_from_displacement, prepend=0)

    apnea_changes = np.where(apnea_diff)[0]
    if apnea_from_displacement[0] == -1:
        apnea_changes = apnea_changes[1:]
    apnea_duration = apnea_changes[1::2] - apnea_changes[::2]  # apneas[:, 1]
    apnea_idx = apnea_changes[::2]  # np.where(apnea_duration != 'missing')
    apnea_end_idx = apnea_changes[1::2]
    apnea_type = apnea_from_displacement[apnea_idx]  # apneas[:, 2]

    apneas_mask = np.zeros(len(respiration))

    for a_idx, start_idx in enumerate(apnea_idx):
        if apnea_type[a_idx] not in apnea_class.values():
            print(apnea_type[a_idx], "not in apnea_class.values()")
            continue
        if float(apnea_duration[a_idx]) == 0.0:
            continue
        end_idx = apnea_end_idx[a_idx]
        apneas_mask[start_idx:end_idx] = 1

        apnea_segments.append([start_idx, end_idx, apnea_duration, apnea_type[a_idx]])
    return apnea_segments


def getSegments(v):
    #

    segments = []

    v_diff = np.diff(v, prepend=v[0])

    v_changes = np.where(v_diff)[0]
    if len(v_changes) % 2:
        v_changes = np.append(v_changes, len(v))

    v_idx = v_changes[::2]  # np.where(apnea_duration != 'missing')
    v_end_idx = v_changes[1::2]


    for a_idx, start_idx in enumerate(v_idx):
        # if float(v_duration[a_idx]) == 0.0:
        #     continue
        end_idx = v_end_idx[a_idx]
        segments.append([start_idx, end_idx])
    return segments



def get_bins_nw(base_path, setup):
    bins_fn = str(setup)+'_estimated_range.npy'

    try:
        bins = np.load(os.sep.join([base_path, bins_fn]), allow_pickle=True)
    except:
        return None

    return bins

def get_ec_reason(base_path, setup):
    bins_fn = str(setup)+'_reject_reason.npy'

    try:
        bins = np.load(os.sep.join([base_path, bins_fn]), allow_pickle=True)
    except:
        return None

    return bins

def get_autocorr_data(base_path, setup):
    bins_fn = str(setup)+'acc_autocorr.npy'

    try:
        bins = np.load(os.sep.join([base_path, bins_fn]), allow_pickle=True)
    except:
        return None

    return bins

gt_dict = { 108139:6, 108145:63, 108146:2, 108147:32, 108148:143, 108152:88, 108153:31, 108154:26, 108161:93, 108170:16, 108171: 74, 108175:2, 108186:270}
device_map = {232:1, 238:1, 234:1, 236:1, 231:1,
                  240:2, 248:2, 254:2, 250:2, 251:2,
                  270:3, 268:3, 256:3, 269:3, 259:3,
                    278:4, 279:4, 273:4, 271:4, 274:4}
if __name__ == '__main__':

    res_dict = {}#'256':[], '273':[], '254':[], '234':[]}
    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']


    if setups == 'all':
        setups = []
        setup_files = fnmatch.filter(os.listdir(base_path),'*_X.npy')
        label_files = fnmatch.filter(os.listdir(base_path),'*_y.npy')
        valid_files = fnmatch.filter(os.listdir(base_path),'*_valid.npy')
        ss_files = fnmatch.filter(os.listdir(base_path),'*_ss_ref*.npy')
        empty_files = fnmatch.filter(os.listdir(base_path),'*_empty*.npy')
        apnea_files = fnmatch.filter(os.listdir(base_path),'*_apnea*.npy')

        print("setup_files", setup_files)

        for i, fn in enumerate(setup_files):
            sess = int(fn[0:fn.find('_')])
            setups.append(sess)

    print(setups)
    y_true = []
    y_pred = []
    y_true_4class = []
    y_pred_4class = []
    y_true_2class = []
    y_pred_2class = []

    sessions_processed = []
    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)

        session = db.session_from_setup(sess)
        rejected = [108146, 108152]
        if session in rejected:
            continue
        fig, ax = plt.subplots(4, sharex=False, figsize=(14, 7))
        try:

            #respiration, fs_new, bins = getSetupRespirationCloudDB(sess)
            ph = np.load('/Neteera/Work/homes/dana.shavit/Research/apnea2021/prepared_data/stitched_2611/'+str(sess)+'_phase.npy')
            phase_df = pd.DataFrame(ph)

            phase_df.interpolate(method="linear", inplace=True)
            phase_df.fillna(method="bfill", inplace=True)
            phase_df.fillna(method="pad", inplace=True)

            ph = phase_df.to_numpy()
            respiration = compute_respiration(ph.flatten())

            UP = 1
            DOWN = 50
            respiration = sp.resample_poly(respiration, UP, DOWN)
            fs_new = int((db.setup_fs(sess) * UP) / DOWN)

        except:
            continue
        print("setup length", len(respiration)/fs_new)
        min_setup_length = 900
        if len(respiration) / fs_new < min_setup_length:
            continue

        model_path = os.path.join('/Neteera/Work/homes/dana.shavit/work/300622/Vital-Signs-Tracking/pythondev/', 'NN', 'apnea')

        json_fn = '6284_model.json'
        hdf_fn = '6284_model.hdf5'
        json_file = open(os.path.join(model_path, json_fn), 'r')
        model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(model_json)
        model.load_weights(os.path.join(model_path, hdf_fn))
        model.compile(loss='huber_loss', optimizer=keras.optimizers.Adam())

        X_fn = [f for f in setup_files if str(sess) in f][0]
        y_fn = [f for f in label_files if str(sess) in f][0]
        v_fn = [f for f in valid_files if str(sess) in f][0]
        e_fn = [f for f in empty_files if str(sess) in f][0]
        a_fn = [f for f in apnea_files if str(sess) in f][0]
        s_fn = [f for f in ss_files if str(sess) in f][0]
        try:
            X = np.load(os.path.join(base_path, X_fn), allow_pickle=True)
            y = np.load(os.path.join(base_path, y_fn), allow_pickle=True)
            valid = np.load(os.path.join(base_path, v_fn), allow_pickle=True)
            ss = np.load(os.path.join(base_path, s_fn), allow_pickle=True)
            empty = np.load(os.path.join(base_path, e_fn), allow_pickle=True)
            apnea = np.load(os.path.join(base_path, a_fn), allow_pickle=True)
            print(len(X), len(y), len(valid), len(ss), len(empty))

        except:
            print("failed to load setup data")
            continue

        session = db.session_from_setup(sess)
        print(session, sess)
        preds = model.predict(X).flatten()
        preds_int = np.stack([int(p) for p in preds])
        print(y)
        print(preds_int)
        #print(np.round(preds.flatten()[valid == 1]))
        pi=0
        time_chunk = 1200
        # for i in range(time_chunk, len(respiration), time_chunk):
        #     if pi > len(preds)-1:
        #         break
        #     pred = preds[pi][0]
        #     ax[1].axvspan(i-time_chunk, i,color='green', alpha=0.6*valid[pi])
        #     ax[0].axvspan(i-time_chunk, i,color='red', alpha=pred/30)
        #
        #     pi += 1
        n_apneas_y = np.sum(y)
        n_apneas_pdf = 0# gt_dict[session]
        n_apneas_pred = np.sum(preds)
        n_apneas_pred_valid = np.sum(preds[valid == 1])

        print("#apneas",np.sum(y), np.sum(preds))
        print("full %", len(empty[empty==1])/len(empty),  len(empty[empty==1]),len(empty))
        print("sleep %", len(ss[ss==1])/len(ss),  len(ss[ss==1]),len(ss))
        print("#apneas(valid)", np.sum(y[valid==True]), np.sum(preds[valid==True]))
        # for i in range(len(preds)):
        #     label = y[i]
        #     pred = preds[i][0]
        #     print(i,label, pred, valid[i])
        #     ax[1].scatter(label, pred, s=3, color='green', alpha=1 if valid[i] else 0.3)
        ax[0].plot(y, linewidth=0.75)
        ax[0].plot(preds, linewidth=0.75)

        preds[valid == False] = 0
        y[valid == False] = 0
        ax[1].plot(y, linewidth=0.75)
        ax[1].plot(preds, linewidth=0.75)
        ax[1].axhline(y=5, color='red', linewidth=0.5)
        ax[1].axhline(y=10, color='red', linewidth=0.5)
        ax[1].axhline(y=15, color='red', linewidth=0.5)
        ax[0].axhline(y=5, color='red', linewidth=0.5)
        ax[0].axhline(y=10, color='red', linewidth=0.5)
        ax[0].axhline(y=15, color='red', linewidth=0.5)
        ax[2].plot(respiration, linewidth=0.5)
        ss = np.repeat(ss, fs_new)
        apnea = apnea * np.max(respiration)
        apnea = np.repeat(apnea, fs_new)
        empty = np.repeat(empty, fs_new)
        ax[3].plot(empty, c='red', alpha=0.5, linewidth=0.5)
        ax[3].plot(ss, c='blue', alpha=0.5, linewidth=0.5)
        ax[2].plot(apnea, c='blue', alpha=0.5, linewidth=0.5)

        device_id = db.setup_sn(sess)[0]
        device_location = device_map[int(device_id) % 1000]
        ax[0].set_title(str(session)+" "+str(sess)+" "+str(device_location))

        ahi = np.mean(preds)*4

        sleep_perc = np.round(len(valid[valid == 1])/len(valid),2)

        if session not in res_dict.keys():
            res_dict[session] = []

        print("LENGTH", len(respiration)/fs_new, len(apnea), len(ss), len(empty))

        duration = len(respiration)/36000
        res_dict[session].append([sess, db.setup_sn(sess)[0][-3:], duration, int(n_apneas_y/duration), int(n_apneas_pred/duration/sleep_perc), int(n_apneas_pred_valid/duration/sleep_perc), sleep_perc])
        print(session, [sess, db.setup_sn(sess)[0][-3:], duration, int(n_apneas_y/duration), int(n_apneas_pred/duration/sleep_perc), int(n_apneas_pred_valid/duration/sleep_perc), sleep_perc])
        outfolder = base_path
        plt.savefig(base_path + str(session) + "_" + str(sess) +'_pred.png', dpi=300)

        if session == 108168:
            plt.show()
        plt.close()
        #continue
        gt = n_apneas_y/duration
        p = n_apneas_pred_valid/duration/sleep_perc
        y_true.append(gt)
        y_pred.append(p)

        def ahi_class(ahi):
            if ahi < 5:
                return 0
            if ahi <= 15:
                return 1
            if ahi <= 30:
                return 2
            return 3

        y_true_4class.append(ahi_class(gt))
        y_pred_4class.append(ahi_class(p))
        y_true_2class.append([0 if ahi_class(gt) <= 1 else 1])
        y_pred_2class.append([0 if ahi_class(p) <= 1 else 1])

    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    plt.figure()
    cm4 = confusion_matrix(y_pred_4class, y_true_4class)
    cm2 = confusion_matrix(y_pred_2class, y_true_2class)
    bx = sns.heatmap(cm4, annot=True, fmt="d", cmap="rocket")

    # Add labels and title
    bx.invert_yaxis()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('4-class Confusion Matrix')
    plt.savefig(os.path.join(base_path,"cm4.png"))
    # Show the plot
    plt.show()

    bx = sns.heatmap(cm2, annot=True, fmt="d", cmap="rocket")
    bx.invert_yaxis()
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('2-class Confusion Matrix')
    plt.savefig(os.path.join(base_path,"cm2.png"))
    plt.show()
    device_map = {232: 1, 238: 1, 234: 1, 236: 1, 231: 1,
                  240: 2, 248: 2, 254: 2, 250: 2, 251: 2,
                  270: 3, 268: 3, 256: 3, 269: 3, 259: 3,
                  278: 4, 279: 4, 273: 4, 271: 4, 274: 4}
    plt.figure(figsize=(10,10))

    #sess, db.setup_sn(sess)[0][-3:], db.setup_duration(sess), 3n_apneas_pdf,4 n_apneas_y, 5n_apneas_pred, 6n_apneas_pred_valid, sleep_perc])
    print("No Valid")


    for k,vv in res_dict.items():
        # print(k,vv)

        for v in vv:

            device_loc = device_map[int(v[1])]

            try:
                plt.scatter(int(v[3]), int(v[4]), alpha=0.8, c=color_radar[device_loc],  s=5)#, v[1], fontsize=5
                #plt.scatter(int(v[5]), int(v[3]), alpha=0.8, c='blue',  s=3)#, v[1], fontsize=5
                if device_loc == 2:
                    al = 0.3
                    fs=6
                else:
                    al=1
                    fs=6

                #plt.text(int(v[5]), int(v[3]), str(v[0])+' '+str(k)[-3:]+' '+str(v[7])+' '+str(device_loc), fontsize=fs, alpha=al,c=color_radar[device_loc])#, v[1]
                #plt.text(int(v[3]), int(v[5]), str(v[0])+' '+str(v[2])+' '+str(v[7])+' '+str(device_loc), fontsize=fs, alpha=al,c=color_radar[device_loc])#, v[1]
                plt.text(int(v[3]), int(v[4]), str(device_loc)+' '+str(k), fontsize=fs, alpha=0.6,c=color_radar[device_loc])#, v[1]
            except:
                print(1)
    th = [5,15,30]
    for t in th:
        thick = 0.5
        if t == 15:
            thick = 1
        plt.axhline(y=t, color='grey', linewidth=thick, alpha=thick)
        plt.axvline(x=t, color='grey', linewidth=thick, alpha=thick)
    plt.title("MB2 AHI PREDICTION NO VALID "+str(len(res_dict.items()))+" Patients Processed")
    plt.figure(figsize=(10, 10))
    print("With Valid")
    for k, vv in res_dict.items():
        print(k, vv)

        for v in vv:

            device_loc = device_map[int(v[1])]

            try:
                plt.scatter(int(v[3]), int(v[5]), alpha=0.8, c=color_radar[device_loc], s=5)  # , v[1], fontsize=5
                #plt.scatter(int(v[6]), int(v[4]), alpha=0.8, c='blue', s=3)  # , v[1], fontsize=5
                if device_loc == 2:
                    al = 0.3
                    fs=6
                else:
                    al=1
                    fs=6
                #plt.text(int(v[3]), int(v[6]), str(v[0]) + ' ' + str(v[2])+' '+str(v[7])+' '+str(device_loc), fontsize=fs, alpha=al,c=color_radar[device_loc])  # , v[1]
                plt.text(int(v[3]), int(v[5]), str(device_loc)+' '+str(k), fontsize=fs, alpha=0.6,c=color_radar[device_loc])  # , v[1]
                #plt.text(int(v[6]), int(v[4]), str(v[0]) + ' ' + str(k)[-3:]+' '+str(v[7])+' '+str(device_loc), fontsize=fs, alpha=al,c=color_radar[device_loc])  # , v[1]
            except:
                print(2)
    th = [5, 15, 30]
    for t in th:
        thick=0.5
        if t == 15:
            thick = 1
        plt.axhline(y=t, color='grey', linewidth=thick, alpha=thick)
        plt.axvline(x=t, color='grey', linewidth=thick, alpha=thick)
    plt.xlabel("gt")
    plt.ylabel("pred")
    plt.title("MB2 AHI PREDICTION WITH VALID "+str(len(res_dict.items()))+" Patients Processed")

    plt.show()

