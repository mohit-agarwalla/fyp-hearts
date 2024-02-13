import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import re
import pickle
from tqdm import tqdm
import os
import ast
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, auc
import glob
import time
import tensorflow as tf
import torch
import random

NOISE_DB_PATH = "C:\Mohit\Imperial\FYP - Local\\fyp-hearts\datasets\mit-bih-noise-stress-test-database-1.0.0"
PTB_PATH = "C:\Mohit\Imperial\FYP - Local\\fyp-hearts\datasets\PTB-XL"

def load_record(record_name: str, database: str):
    """
    Read an ECG record based on the database that it is from.
    ----------------------------------------------------------------
    Parameters:
    - record_name: The name of the record from each database.
    - database: The name of the database. Used to determine which path to use.
    
    Returns:
    - record: Dataframe containing ecg record 
    """
    
    # Determine path based on database
    if database.lower() == "ptb":
        path = PTB_PATH
    elif database.lower() == "noise":
        path = NOISE_DB_PATH
    
    if record_name[0] != "\\":
        record_name = "\\" + record_name
    
    # Modulate path to be real path
    path = path + record_name
    
    # Find record and read into dataframe
    record = wfdb.rdrecord(path).to_dataframe()
    
    return record
    

def generate_ecg_with_noise(ecg_dataframe, noise_dataframe, snr):
    """
    Generate an ECG signal with a specified signal-to-noise ratio (SNR).

    Parameters:
    - ecg_dataframe: DataFrame containing the ECG signal.
    - noise_dataframe: DataFrame containing the noise signal.
    - snr: Signal-to-noise ratio in dB.

    Returns:
    - ecg_with_noise: DataFrame containing the generated ECG signal with added noise.
    """

    # Extract ECG and noise signals from the dataframes
    ecg_signal = ecg_dataframe['ECG_Signal'].values
    noise_signal = noise_dataframe['Noise_Signal'].values

    # Calculate the power of the original ECG signal
    ecg_power = np.mean(ecg_signal ** 2)

    # Calculate the noise power required for the specified SNR
    noise_power = ecg_power / (10 ** (snr / 10))

    # Scale the noise signal to match the desired power
    scaled_noise = np.sqrt(noise_power) * (noise_signal / np.std(noise_signal))

    # Add the scaled noise to the original ECG signal
    ecg_with_noise = ecg_signal + scaled_noise

    # Create a new DataFrame with the generated ECG signal
    ecg_with_noise_df = pd.DataFrame({'ECG_With_Noise': ecg_with_noise})
    
    return ecg_with_noise_df

def real_accuracy_score(y_true, y_pred):
    acc = accuracy_score(y_true.values.flatten(), y_pred.flatten())
    return acc

def numeric(df, cols=None):
    """
    Take every column of a dataframe and make numeric
    """
    if cols == None:
        cols = df.columns
    
    for col in cols:
        df[col] = pd.to_numeric(df[col])

def read_green_ppg_raw(filename):
    """
    Take the filename of a green PPG file and return a dataframe.
    ------------------------------------------------
    
    Parameters:
        filename: (str) Full filename of the green PPG file.
    
    Returns:
        ppg_df: (pd.DataFrame) Dataframe of a green PPG file.
    """
    # Open file and read file's contents
    with open(filename, "r") as f:
        file_contents = f.read()
    
    # Define regex to extract values
    pattern = re.compile(r"PpgGreenTimestamp : (\d+) : PpgGreenValue : (\d+)")
    
    # Find matches in the string
    matches = pattern.findall(file_contents)
    
    # Create a DataFrame from the extracted values
    ppg_df = pd.DataFrame(matches, columns=['PpgGreenTimestamp', 'PpgGreenValue'])
    
    return ppg_df

def load_raw_data_ptbxl(df, sampling_rate, path):
    if os.path.exists(path + f'raw{sampling_rate}.npy'):
        data = np.load(path+f'raw{sampling_rate}.npy', allow_pickle=True)
    else:
        if sampling_rate==100:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
        
        data = np.array([signal for signal, meta in data])
        pickle.dump(data, open(path+f'raw{sampling_rate}.npy', 'wb'))
    
    return data

def load_dataset(path, sampling_rate, release=False):
    if 'PTB-XL' in path:
        # load data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)
        
        # # Remove data of other leads if looking for single lead only
        # if single_lead:
        #     X = X[:,:,0]
        
    return X, Y

def compute_label_aggregations(df, folder, ctype):
    # How many diagnostic scp codes in each row
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    # Find the codes from the scp_statements folder 
    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)
    
    # if looking at a specific task
    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def select_subset(raw_labels, data, subset=['age', 'weight', 'sex', 'height']):
    nan_indices = np.where(pd.isnull(raw_labels[subset]))[0]
    new_data = np.delete(data, nan_indices, axis=0)
    labels = raw_labels.dropna(subset=subset)
    return labels, new_data

def select_data(XX,YY, ctype, min_samples, output_folder):
    # Convert multilabel to multi-hot encoder
    mlb = MultiLabelBinarizer()
    
    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
        
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    
    elif ctype == 'form':
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    
    elif ctype == 'rhythm':
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    
    elif ctype == 'all':
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    
    else:
        pass
    
    # Save the local binarizer
    with open(output_folder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)
    
    return X, Y, y, mlb

def select_data_demo(XX,YY, ctype, min_samples, output_folder, demographics):
    # Convert multilabel to multi-hot encoder
    mlb = MultiLabelBinarizer()
    
    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
        
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        demographics = demographics[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    
    
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        demographics = demographics[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    
    elif ctype == 'form':
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        demographics = demographics[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    
    elif ctype == 'rhythm':
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        demographics = demographics[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    
    elif ctype == 'all':
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        demographics = demographics[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    
    else:
        pass
    
    # Save the local binarizer
    with open(output_folder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)
    
    return X, Y, y, demographics, mlb

def apply_standard_scaler(X, ss):
    X_tmp = []
    
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

def preprocess_signals(X_train, X_val, X_test, output_folder):
    # Standardise for mu=0 and variance=1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    
    with open(output_folder+'standard_scaler.pkl', 'wb') as ss_file:
        pickle.dump(ss, ss_file)
    
    X_train = apply_standard_scaler(X_train, ss)
    X_test = apply_standard_scaler(X_test, ss)
    X_val = apply_standard_scaler(X_val, ss)
    
    return X_train, X_val, X_test

def get_appropriate_bootstrap_samples(y_true, n_bootstrap_samples):
    samples = []
    while True:
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstrap_samples:
                break
    
    return samples

def generate_results(idxs, y_true, y_pred, thresholds):
    return evaluate_experiment(y_true[idxs], y_pred[idxs], thresholds)

def evaluate_experiment(y_true, y_pred, thresholds):
    results = {}
    if not thresholds is None:
        # binary predictions
        # Need to add when necessary
        pass
    
    # Would be useful to create 
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    
    return pd.DataFrame(results, index=[0])

def generate_summary_table(selection=None, exps=None, folder='output/'):
    if exps is None:
        exps = ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']
    metric1 = 'macro_auc'
    
    # getmodels
    models = {}
    for i,exp in enumerate(exps):
        if selection is None:
            exp_models = [m.split('/')[-1] for m in glob.glob(f'{folder}{exp}/models/*')]
            print(exp_models)
        else:
            exp_models = selection
        if i==0:
            models = set(exp_models)
        else:
            models = models.union(set(exp_models))
    
    results_dic = {'Method': []}
    for exp in exps:
        results_dic[f'{exp}_AUC'] = []
    
    for model in models:
        results_dic['Method'].append(model)
        
        for e in exps:
            try:
                me_res = pd.read_csv(folder+str(e)+'/models/'+str(model)+'/results/te_results.csv', index_col=0)
                
                mean1 = me_res.loc['point'][metric1]
                unc1 = max(me_res.loc['upper'][metric1]-me_res.loc['point'][metric1], me_res.loc['point'][metric1]-me_res.loc['lower'][metric1])
                
                results_dic[e+'_AUC'].append("%.3f(%.2d)" %(np.round(mean1,3), int(unc1*1000)))
            except FileNotFoundError:
                results_dic[e+'_AUC'].append("---")
    
    df = pd.DataFrame(results_dic)
    df_index = df[df.Method.isin(['naive', 'ensemble'])]
    df_rest = df[~df.Method.isin(['naive', 'ensemble'])]
    df = pd.concat([df_rest, df_index])
    df.to_csv(folder+'results_ptbxl.csv')
    
    print(df)
    return df

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def get_appropriate_bootstrap(y_true, n_bootstrapping_samples):
    samples = []
    while True:
        ridxs = np.random.randint(0,len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstrapping_samples:
                break
    
    return samples

def aggregate_predictions(preds,targs=None,idmap=None,aggregate_fn = np.mean,verbose=True):
    '''
    aggregates potentially multiple predictions per sample (can also pass targs for convenience)
    idmap: idmap as returned by TimeSeriesCropsDataset's get_id_mapping
    preds: ordered predictions as returned by learn.get_preds()
    aggregate_fn: function that is used to aggregate multiple predictions per sample (most commonly np.amax or np.mean)
    '''
    if(idmap is not None and len(idmap)!=len(np.unique(idmap))):
        if(verbose):
            print("aggregating predictions...")
            preds_aggregated = []
            targs_aggregated = []
            for i in np.unique(idmap):
                preds_local = preds[np.where(idmap==i)[0]]
                preds_aggregated.append(aggregate_fn(preds_local,axis=0))
                if targs is not None:
                    targs_local = targs[np.where(idmap==i)[0]]
                    assert(np.all(targs_local==targs_local[0])) #all labels have to agree
                    targs_aggregated.append(targs_local[0])
            if(targs is None):
                return np.array(preds_aggregated)
            else:
                return np.array(preds_aggregated),np.array(targs_aggregated)
    else:
        if(targs is None):
            return preds
        else:
            return preds,targs
        
def valid_labels(task,labels):
    if task == 'all':
        labels = labels[labels.all_scp_len > 0]
    elif task == 'rhythm':
        labels = labels[labels.rhythm_len > 0]
    elif task == 'form':
        labels = labels[labels.form_len > 0]
    elif task == 'superdiagnostic':
        labels = labels[labels.superdiagnostic_len > 0]
    elif task == 'subdiagnostic':
        labels = labels[labels.subdiagnostic_len > 0]
    elif task == 'diagnostic':
        labels = labels[labels.diagnostic_len > 0]
    
    return labels
        

def analyse_by_sex(experiment, models, datafolder, outputfolder, sampling_rate=100):
    name, task = experiment[0], experiment[1]
    # Find relevant data to determine whether to contain or not contain
    _, raw_labels = load_dataset(datafolder, sampling_rate=sampling_rate)
    labels = compute_label_aggregations(raw_labels, datafolder, task)
    test_labels = valid_labels(task, labels)
    test_labels = test_labels[test_labels.strat_fold==10] # only consider test folds
    print(test_labels.shape)

    # idxs where male/female
    test_labels_male = np.where(test_labels.sex==0)   
    test_labels_female = np.where(test_labels.sex==1)
    
    print(name)
    auc_exp = {}
    
    for model in models:
        # load y_test and y_pred_test
        modelname = model['modelname']
        if 'demo' in outputfolder:
            y_path = outputfolder+"\\experiments\\"+name+"\\data\y_test.npy"
            y_pred_path = outputfolder+"\\experiments\\"+name+"\\models\\"+modelname+"\\y_test_pred.npy"
        else:
            y_path = outputfolder+"\\"+name+"\\data\y_test.npy"
            y_pred_path = outputfolder+"\\"+name+"\\models\\"+modelname+"\\y_test_pred.npy"
        
        y = np.load(y_path, allow_pickle=True)
        y_pred = np.load(y_pred_path, allow_pickle=True)
        print(modelname, y.shape, y_pred.shape)
        y_male = y[test_labels_male]
        y_male = y_male[:,[i for i in range(y_male.shape[1]) if len(np.unique(y_male[:,i])) == 2]]
        y_female = y[test_labels_female]
        y_female = y_female[:,[i for i in range(y_female.shape[1]) if len(np.unique(y_female[:,i])) == 2]]
        
        y_pred_male = y_pred[test_labels_male]
        y_pred_male = y_pred_male[:,[i for i in range(y_male.shape[1]) if len(np.unique(y_male[:,i])) == 2]]
        y_pred_female = y_pred[test_labels_female]
        y_pred_female = y_pred_female[:,[i for i in range(y_female.shape[1]) if len(np.unique(y_female[:,i])) == 2]]
        
        auc_male = evaluate_experiment(y_male, y_pred_male, thresholds=None).values[0][0]
        auc_female = evaluate_experiment(y_female, y_pred_female, thresholds=None).values[0][0]
        auc_exp[modelname] = [auc_male, auc_female]
    
    print(auc_exp)
    return auc_exp
            
            
            
            


class TimeseriesDatasetCrops(torch.utils.data.Dataset):
    """timeseries dataset with partial crops."""

    def __init__(self, df, output_size, chunk_length, min_chunk_length, memmap_filename=None, npy_data=None, random_crop=True, data_folder=None, num_classes=2, copies=0, col_lbl="label", stride=None, start_idx=0, annotation=False, transforms=[]):
        """
        accepts three kinds of input:
        1) filenames pointing to aligned numpy arrays [timesteps,channels,...] for data and either integer labels or filename pointing to numpy arrays[timesteps,...] e.g. for annotations
        2) memmap_filename to memmap for data [concatenated,...] and labels- label column in df corresponds to index in this memmap
        3) npy_data [samples,ts,...] (either path or np.array directly- also supporting variable length input) - label column in df corresponds to sampleid
        
        transforms: list of callables (transformations) (applied in the specified order i.e. leftmost element first)
        """
        assert not((memmap_filename is not None) and (npy_data is not None))
        #require integer entries if using memmap or npy
        assert (memmap_filename is None and npy_data is None) or df.data.dtype==np.int64
                        
        self.timeseries_df = df
        self.output_size = output_size
        self.data_folder = data_folder
        self.transforms = transforms
        self.annotation = annotation
        self.col_lbl = col_lbl

        self.c = num_classes

        self.mode="files"
        self.memmap_filename = memmap_filename
        if(memmap_filename is not None):
            self.mode="memmap"
            memmap_meta = np.load(memmap_filename.parent/(memmap_filename.stem+"_meta.npz"))
            self.memmap_start = memmap_meta["start"]
            self.memmap_shape = tuple(memmap_meta["shape"])
            self.memmap_length = memmap_meta["length"]
            self.memmap_dtype = np.dtype(str(memmap_meta["dtype"]))
            self.memmap_file_process_dict = {}
            if(annotation):
                memmap_meta_label = np.load(memmap_filename.parent/(memmap_filename.stem+"_label_meta.npz"))
                self.memmap_filename_label = memmap_filename.parent/(memmap_filename.stem+"_label.npy")
                self.memmap_shape_label = tuple(memmap_meta_label["shape"])
                self.memmap_file_process_dict_label = {}
                self.memmap_dtype_label = np.dtype(str(memmap_meta_label["dtype"]))
        elif(npy_data is not None):
            self.mode="npy"
            if(isinstance(npy_data,np.ndarray) or isinstance(npy_data,list)):
                self.npy_data = np.array(npy_data)
                assert(annotation is False)
            else:
                self.npy_data = np.load(npy_data)
            if(annotation):
                self.npy_data_label = np.load(npy_data.parent/(npy_data.stem+"_label.npy"))
        
        self.random_crop = random_crop

        self.df_idx_mapping=[]
        self.start_idx_mapping=[]
        self.end_idx_mapping=[]

        for df_idx,(id,row) in enumerate(df.iterrows()):
            if(self.mode=="files"):
                data_length = row["data_length"]
            elif(self.mode=="memmap"):
                data_length= self.memmap_length[row["data"]]
            else: #npy 
                data_length = len(self.npy_data[row["data"]])
                                              
            if(chunk_length == 0):#do not split
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(start_idx,data_length,chunk_length if stride is None else stride))
                idx_end = [min(l+chunk_length, data_length) for l in idx_start]

            #remove final chunk(s) if too short
            for i in range(len(idx_start)):
                if(idx_end[i]-idx_start[i]< min_chunk_length):
                    del idx_start[i:]
                    del idx_end[i:]
                    break
            #append to lists
            for _ in range(copies+1):
                for i_s,i_e in zip(idx_start,idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
                    
    def __len__(self):
        return len(self.df_idx_mapping)

    def __getitem__(self, idx):
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        #determine crop idxs
        timesteps= end_idx - start_idx
        assert(timesteps>=self.output_size)
        if(self.random_crop):#random crop
            if(timesteps==self.output_size):
                start_idx_crop= start_idx
            else:
                start_idx_crop = start_idx + random.randint(0, timesteps - self.output_size -1)#np.random.randint(0, timesteps - self.output_size)
        else:
            start_idx_crop = start_idx + (timesteps - self.output_size)//2
        end_idx_crop = start_idx_crop+self.output_size

        #print(idx,start_idx,end_idx,start_idx_crop,end_idx_crop)
        #load the actual data
        if(self.mode=="files"):#from separate files
            data_filename = self.timeseries_df.iloc[df_idx]["data"]
            if self.data_folder is not None:
                data_filename = self.data_folder/data_filename
            data = np.load(data_filename)[start_idx_crop:end_idx_crop] #data type has to be adjusted when saving to npy
            
            ID = data_filename.stem

            if(self.annotation is True):
                label_filename = self.timeseries_df.iloc[df_idx][self.col_lbl]
                if self.data_folder is not None:
                    label_filename = self.data_folder/label_filename
                label = np.load(label_filename)[start_idx_crop:end_idx_crop] #data type has to be adjusted when saving to npy
            else:
                label = self.timeseries_df.iloc[df_idx][self.col_lbl] #input type has to be adjusted in the dataframe
        elif(self.mode=="memmap"): #from one memmap file
            ID = self.timeseries_df.iloc[df_idx]["data_original"].stem
            memmap_idx = self.timeseries_df.iloc[df_idx]["data"] #grab the actual index (Note the df to create the ds might be a subset of the original df used to create the memmap)
            idx_offset = self.memmap_start[memmap_idx]
            
            pid = os.getpid()
            #print("idx",idx,"ID",ID,"idx_offset",idx_offset,"start_idx_crop",start_idx_crop,"df_idx", self.df_idx_mapping[idx],"pid",pid)
            mem_file = self.memmap_file_process_dict.get(pid, None)  # each process owns its handler.
            if mem_file is None:
                #print("memmap_shape", self.memmap_shape)
                mem_file = np.memmap(self.memmap_filename, self.memmap_dtype, mode='r', shape=self.memmap_shape)
                self.memmap_file_process_dict[pid] = mem_file
            data = np.copy(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            #print(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            if(self.annotation):
                mem_file_label = self.memmap_file_process_dict_label.get(pid, None)  # each process owns its handler.
                if mem_file_label is None:
                    mem_file_label = np.memmap(self.memmap_filename_label, self.memmap_dtype, mode='r', shape=self.memmap_shape_label)
                    self.memmap_file_process_dict_label[pid] = mem_file_label
                label = np.copy(mem_file_label[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            else:
                label = self.timeseries_df.iloc[df_idx][self.col_lbl]
        else:#single npy array
            ID = self.timeseries_df.iloc[df_idx]["data"]
            
            data = self.npy_data[ID][start_idx_crop:end_idx_crop]
            
            if(self.annotation):
                label = self.npy_data_label[ID][start_idx_crop:end_idx_crop]
            else:
                label = self.timeseries_df.iloc[df_idx][self.col_lbl]
        sample = {'data': data, 'label': label, 'ID':ID}
        
        for t in self.transforms:
            sample = t(sample)

        return sample
    
    def get_sampling_weights(self, class_weight_dict,length_weighting=False, group_by_col=None):
        assert(self.annotation is False)
        assert(length_weighting is False or group_by_col is None)
        weights = np.zeros(len(self.df_idx_mapping),dtype=np.float32)
        length_per_class = {}
        length_per_group = {}
        for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
            label = self.timeseries_df.iloc[i][self.col_lbl]
            weight = class_weight_dict[label]
            if(length_weighting):
                if label in length_per_class.keys():
                    length_per_class[label] += e-s
                else:
                    length_per_class[label] = e-s
            if(group_by_col is not None):
                group = self.timeseries_df.iloc[i][group_by_col]
                if group in length_per_group.keys():
                    length_per_group[group] += e-s
                else:
                    length_per_group[group] = e-s
            weights[iw] = weight

        if(length_weighting):#need second pass to properly take into account the total length per class
            for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
                label = self.timeseries_df.iloc[i][self.col_lbl]
                weights[iw]= (e-s)/length_per_class[label]*weights[iw]
        if(group_by_col is not None):
            for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
                group = self.timeseries_df.iloc[i][group_by_col]
                weights[iw]= (e-s)/length_per_group[group]*weights[iw]

        weights = weights/np.min(weights)#normalize smallest weight to 1
        return weights

    def get_id_mapping(self):
        return self.df_idx_mapping
