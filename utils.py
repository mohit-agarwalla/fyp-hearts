import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score
import re
import pickle
from tqdm import tqdm
import os
import ast
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, auc
import glob

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

def load_dataset(path, sampling_rate, single_lead=True, release=False):
    if 'PTB-XL' in path:
        # load data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)
        
        # Remove data of other leads if looking for single lead only
        if single_lead:
            X = X[:,:,0]
        
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
        exps = ['exp3']
    metric1 = 'macro_auc'
    
    # getmodels
    models = {}
    for i,exp in enumerate(exps):
        if selection is None:
            exp_models = [m.split('/')[-1] for m in glob.glob(f'{folder}{exp}/models/*')]
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