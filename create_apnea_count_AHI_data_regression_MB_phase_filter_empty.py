# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import fnmatch

from sklearn import preprocessing
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
from Tests.NN.create_apnea_count_AHI_data import delays, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebug, getSetupRespirationCloudDB, getApneaSegments
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import apnea_class

db = DB()

def create_AHI_regression_training_data_from_annotation(respiration, apnea_ref, time_chunk, step, scale, fs):
    X = []
    y = []
    valid = []
    apnea_ref['start_t'] = fs*apnea_ref['onset']
    apnea_ref['end_t'] = apnea_ref['start_t'] + apnea_ref['duration']*fs
    for i in range(time_chunk, len(respiration), step):

        v = 1
        seg = respiration[i - time_chunk:i]
        if (seg == -100).any():
            v = 0
        if np.mean(seg) < 1e-4 and np.std(seg) < 1e-5:
            v = 0
            #continue
        if len(seg) != time_chunk:
            continue
        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        #print(np.mean(seg), np.median(seg), np.std(seg))
        #print(apneas_df.loc[(apneas_df['start_t'] >= (i - time_chunk)) & (apneas_df['end_t'] <i)])
        num_apneas = len(apnea_ref.loc[(apnea_ref['start_t'] >= (i - time_chunk)) & (apnea_ref['end_t'] <i)])
        #num_apneas = count_apneas_in_chunk(start_t=i - time_chunk, end_t=i, apnea_segments=apnea_segments)

        y.append(num_apneas)

        # plt.plot(preprocessing.robust_scale(seg))
        # plt.title(str(num_apneas))
        # plt.show()

        valid.append(v)
    if len(X):
        print(y)
        X = np.stack(X)
        y = np.stack(y)

        valid = np.stack(valid)
        #print(np.count_nonzero(valid))
        return X, y, valid

    return X,y, valid



def create_AHI_regression_training_data_MB_phase(respiration, apnea_ref, empty_seconds, time_chunk, step, scale, fs):
    print("in")
    X = []
    y = []
    valid = []
    apnea_segments = []
    if apnea_ref is not None:
        #empty_seconds = [max(empty_seconds[i], empty_bins[i]) for i in range(len(empty_seconds))]

        if isinstance(apnea_ref, pd.core.series.Series):
            apnea_ref = apnea_ref.to_numpy()

        apnea_diff = np.diff(apnea_ref, prepend=0)

        apnea_changes = np.where(apnea_diff)[0]
        apnea_duration = apnea_changes[1::2] - apnea_changes[::2]  # apneas[:, 1]
        apnea_idx = apnea_changes[::2]  # np.where(apnea_duration != 'missing')
        apnea_end_idx = apnea_changes[1::2]
        apnea_type = 1  # apneas[:, 2]

        for a_idx, start_idx in enumerate(apnea_idx):

            if float(apnea_duration[a_idx]) == 0.0:
                continue
            end_idx = apnea_end_idx[a_idx]

            apnea_segments.append([start_idx, end_idx, apnea_duration, 1    ])
        print(len(apnea_segments), "apneas in setup")

    filter_hour_1 = True
    delay = 60 if filter_hour_1 else 0
    print("delay = ", delay)
    for i in range(time_chunk, len(respiration), step):
    #for i in range(time_chunk, len(respiration), step):
        v = 1

        if i < int(delay*fs):
            v = 0

        seg = respiration[i - time_chunk:i]
        start_fs = int((i - time_chunk)/fs)
        end_fs = int(i/fs)

        if empty_seconds is not None and end_fs < len(empty_seconds):
            empty_ref = empty_seconds[start_fs:end_fs]
            #print(np.sum(empty_ref)/len(empty_ref))
            if np.sum(empty_ref)/len(empty_ref) > 0.1:
                v = 0
            #print(start_fs, end_fs, np.sum(empty_ref), len(empty_ref), v)
        if (seg == -100).any():
            v = 0
        if np.mean(seg) < 1e-4 and np.std(seg) < 1e-5:
            v = 0
        if len(seg) != time_chunk:
            continue


        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        num_apneas = 0 if apnea_ref is None else count_apneas_in_chunk(start_t=start_fs, end_t=end_fs, apnea_segments=apnea_segments)

        y.append(num_apneas)
        valid.append(v)

    if len(X):
        X = np.stack(X)
        y = np.stack(y)

        valid = np.stack(valid)
        print(np.count_nonzero(valid))
        print(len(X), len(y), np.sum(valid) / len(valid))
        return X, y, valid


    return X,y, valid
#
#

def get_empty_seconds_mb(setup):
    empty = None
    try:
        db.update_mysql_db(setup)
        p = db.setup_dir(setup)
        ref_dir = os.sep.join([p, 'NES_RES'])
        print(ref_dir)
        csv_file = fnmatch.filter(os.listdir(ref_dir), '*_VS.csv')[0]
        df = pd.read_csv(os.sep.join([ref_dir, csv_file]))
        empty = np.zeros(len(df['stat']))
        empty[df['stat'] == 'Empty'] = 1
    except:
        print("something broken in loading res")
    return empty


def load_apnea_ref_from_annotations(setup, db):
    try:
        s = db.session_from_setup(setup)
        apnea = load_reference(setup, 'apnea', db)

        if setup in delays.keys():
            apnea = apnea[delays[setup]:]
        if apnea is not None:
            return apnea

        data_setup = min(db.setups_by_session(s))
        p = db.setup_dir(data_setup)
        ref_dir = os.sep.join([p, 'REFERENCE/RESPIRONICS_ALICE6'])
        #print(ref_dir)
        #print(os.listdir(ref_dir))
        apnea = None
        for file in os.listdir(ref_dir):
            #print(file)
            if 'pnea.npy' in file:
                #print(os.path.join(ref_dir, file))
                anno_path = os.path.join(ref_dir, file)
                #print(anno_path)
                apnea = np.load(anno_path, allow_pickle=True)
                print('loaded apnea, setup', setup)
                break
        if len(apnea) > 0:
            #print(apnea.keys())
            print('ok')
            return apnea
        else:
            print(setup, "not ok, no ref")
            return None
    except:
        print(setup, "not ok exception")
        return None

def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')
 parser.add_argument('-save_path', metavar='Location', type=str, required=True, help='location of saved data')
 parser.add_argument('--scale', action='store_true', help='Scale data to m=0 s=1')
 parser.add_argument('--show', action='store_true', help='Scale data to m=0 s=1')
 parser.add_argument('--overwrite', action='store_true',  required=False, help='Overwrite existing output')
 parser.add_argument('--filter_empty', action='store_true',  required=False, help='Filter empty from input')
 parser.add_argument('-chunk', metavar='window', type=int, required=True, help='signal size')
 parser.add_argument('-step', metavar='window', type=int, required=True, help='stride for signal creation')


 return parser.parse_args()

if __name__ == '__main__':

    args = get_args()

    if args.scale:
        save_path = os.path.join(args.save_path, 'scaled')
    else:
        save_path = os.path.join(args.save_path, 'unscaled')

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    sessions = []
    db = DB('neteera_cloud_mirror')

    setups = MB_HQ

    print(setups)
    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)

        print(":::::::: processing session", sess, str(i_sess)+'/'+str(len(setups)), "::::::::")
        if args.overwrite and os.path.isfile(os.path.join(save_path,str(sess) + '_X.npy')):
            print(sess, "done, skipping")
            continue
        try:
            apnea_reference = load_apnea_ref_from_annotations(sess, db)
            #respiration, fs_new = getSetupRespirationCloudDB(sess)
            if apnea_reference is None:
                print("no reference for setup", sess)

            print(np.unique(apnea_reference))
            apnea_ref_class = np.zeros(len(apnea_reference))

            for i in range(len(apnea_reference)):
                #print(i, apnea_reference[i], apnea_reference[i] in apnea_class.keys())
                if apnea_reference[i] not in apnea_class.keys():
                    apnea_ref_class[i] = -1
                else:
                    apnea_ref_class[i] = int(apnea_class[apnea_reference[i]])

            print(np.unique(apnea_ref_class))

            respiration, fs_new, bins = getSetupRespirationCloudDB(sess)
            min_setup_length = 15000
            if len(respiration)/fs_new < min_setup_length:
                continue

            chunk_size_in_minutes = args.chunk
            time_chunk = fs_new * chunk_size_in_minutes * 60
            step = fs_new * args.step * 60

            empty_ref_mb = None
            if args.filter_empty:
                empty_ref_mb = get_empty_seconds_mb(sess)
                if len(bins) !=len(empty_ref_mb):
                    print("respiration", len(respiration), "empty_sec_mb", len(empty_ref_mb), "bins", len(bins))
                if len(empty_ref_mb) < len(bins):
                    empty_ref_mb =np.pad(empty_ref_mb, [0, len(bins)-len(empty_ref_mb)], mode='constant', constant_values=(0))
                elif len(empty_ref_mb) > len(bins):
                    empty_ref_mb = empty_ref_mb[:len(bins)]
                if len(bins) !=len(empty_ref_mb):
                    print("***")

                empty_ref_mb[bins == -1] = 1


            X, y, valid,  = create_AHI_regression_training_data_MB_phase(respiration=respiration,
                                                                        apnea_ref=apnea_ref_class,
                                                                        empty_seconds=empty_ref_mb,
                                                                        time_chunk=time_chunk,
                                                                        step=step,
                                                                        scale=args.scale,
                                                                        fs=fs_new)

        except:
            print(sess, "not ok 2")
            continue
        print(X.shape, y.shape)
        print("successfully created AHI labels")
        np.save(os.path.join(save_path,str(sess) + '_y.npy'), y, allow_pickle=True)
        #np.save(os.path.join(save_path,str(sess) + '_y3.npy'), y3, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_X.npy'), X, allow_pickle=True)
        print(valid
              )
        np.save(os.path.join(save_path, str(sess) + '_valid.npy'), valid, allow_pickle=True)
        print("saved training data")
