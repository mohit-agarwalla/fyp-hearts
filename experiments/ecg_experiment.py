"""
Adapted from: https://github.com/helme/ecg_ptbxl_benchmarking/blob/bed65591f0e530aa6a9cb4a4681feb49c397bf02/code/experiments/scp_experiment.py
"""

import utils
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
from models.baseline_wavelet import WaveletModel
from models.baseline_resnet import ResNet
from models.classifer import Classifier
# from models.baseline_conv1d import SimpleCNN
from models.baseline_ann import BaselineANN
from models.lstm import LSTM_Model
from tensorflow.keras.callbacks import EarlyStopping


class ECG_Experiment:
    """
    Run standard experiments on ECG datasets
    """
    
    def __init__(self, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat'):
        """
        Parameters:
        experiment_name : str
            The experiment name, used for saving outputs as well.
        task : 
        """
        self.experiment_name = experiment_name
        self.task = task
        self.datafolder = datafolder
        self.outputfolder = outputfolder
        self.models = models
        self.sampling_frequency = sampling_frequency
        self.min_samples = min_samples
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        
        print(self.models)
        
        # if needed create folder structure
        if not os.path.exists(self.outputfolder+self.experiment_name):
            os.makedirs(self.outputfolder+self.experiment_name)
        if not os.path.exists(self.outputfolder+self.experiment_name+'/results/'):
            os.makedirs(self.outputfolder+self.experiment_name+'/results/')
        if not os.path.exists(self.outputfolder+self.experiment_name+'/models/'):
            os.makedirs(self.outputfolder+self.experiment_name+'/models/')
        if not os.path.exists(self.outputfolder+self.experiment_name+'/data/'):
            os.makedirs(self.outputfolder+self.experiment_name+'/data/')
        
        print(self.outputfolder+self.experiment_name+'/data/')
        
    def prepare(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)
        
        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)
        
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')
        # One hot encode relevant data
        self.input_shape = self.data[0].shape
        
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        self.y_train.dump(self.outputfolder + self.experiment_name+ '/data/y_train.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name+ '/data/y_val.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name+ '/data/y_test.npy')

        modelname = 'naive'
        # create most naive predictions via simple mean in training
        mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        if not os.path.exists(mpath+'results/'):
            os.makedirs(mpath+'results/')

        mean_y = np.mean(self.y_train, axis=0)
        np.array([mean_y]*len(self.y_train)).dump(mpath + 'y_train_pred.npy')
        np.array([mean_y]*len(self.y_test)).dump(mpath + 'y_test_pred.npy')
        np.array([mean_y]*len(self.y_val)).dump(mpath + 'y_val_pred.npy')
        
        print("prepared")
    
    def perform(self):
        
        for model_desc in self.models:
            modelname = model_desc['modelname']
            modeltype = model_desc['modeltype']
            modelparams = model_desc['modelparams']
            
            print(f"Model is {model_desc}")
            
            mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
            # Create folder for model outputs
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath+'results/'):
                os.makedirs(mpath+'results/')
            
            n_classes = self.Y.shape[1]
            
            if 'lead1' in modelname:
                X_train = self.X_train[:,:,0].reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                X_test = self.X_test[:,:,0].reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
                X_val = self.X_val[:,:,0].reshape(self.X_val.shape[0], self.X_val.shape[1], 1)
            else:
                X_train = self.X_train
                X_test = self.X_test
                X_val = self.X_val
            
            if 'lead1' in modelname:
                leads = 1
            else:
                leads = 12
            
            # load respective model
            if modeltype == 'WAVELET':
                model = WaveletModel(n_classes, self.sampling_frequency, mpath, **modelparams)
                # fit model
                model.fit(X_train, self.y_train, X_val, self.y_val)
                model.save(self.outputfolder+'last_model.h5')
            
            elif modeltype == 'LSTM':
                model = LSTM_Model(n_classes, self.sampling_frequency, mpath, leads, **modelparams) 
                model.fit(X_train, self.y_train, X_val,self.y_val)
                
                model.model.save(self.outputfolder+'last_model.h5')               
            
            elif modeltype == 'simple':
                model = BaselineANN(n_classes=n_classes, freq=self.sampling_frequency, outputfolder=mpath, **modelparams)
                model.fit(X_train, self.y_train, X_val, self.y_val)
                model.model.save(self.outputfolder+'last_model.h5')
            
            elif modeltype == 'RESNET':
                resnet = ResNet(**modelparams)
                
                
                model = Classifier(model=resnet, input_size=1000, n_classes=n_classes, learning_rate=0.0001, leads=leads)
                model.add_compile()
                es = EarlyStopping(monitor='val_loss', patience=6)
                model.fit(X_train, self.y_train, (X_val,self.y_val))
                model.save(self.outputfolder+'last_model.h5')
                
            elif modeltype == "fastai_model":  
                from models.fastai_model import fastai_model
                model = fastai_model(modelname, n_classes, self.sampling_frequency, mpath,self.input_shape, **modelparams)
                model.fit(self.X_train, self.y_train, self.X_val, self.y_val) 
                
        
            
            # predict and dump
            model.predict(X_train).dump(mpath+'y_train_pred.npy')
            model.predict(X_val).dump(mpath+'y_val_pred.npy')
            model.predict(X_test).dump(mpath+'y_test_pred.npy')
            
        modelname = 'ensemble'
        
        # create ensemble preds by simple mean across all model preds
        ensemblepath = self.outputfolder+self.experiment_name+ '/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(ensemblepath):
            os.makedirs(ensemblepath)
        if not os.path.exists(ensemblepath+'/results/'):
            os.makedirs(ensemblepath+'/results/')
        
        ensemble_train, ensemble_val, ensemble_test = [], [], []
        for model_desc in os.listdir(self.outputfolder+self.experiment_name+'/models/'):
            if model_desc not in ['ensemble', 'naive']:
                mpath = self.outputfolder+self.experiment_name+'/models/'+model_desc+'/'
                ensemble_train.append(np.load(mpath+'y_train_pred.npy', allow_pickle=True))
                ensemble_val.append(np.load(mpath+'y_val_pred.npy', allow_pickle=True))
                ensemble_test.append(np.load(mpath+'y_test_pred.npy', allow_pickle=True))
            
        # dump mean predictions
        np.array(ensemble_train).mean(axis=0).dump(ensemblepath + 'y_train_pred.npy')
        np.array(ensemble_test).mean(axis=0).dump(ensemblepath + 'y_test_pred.npy')
        np.array(ensemble_val).mean(axis=0).dump(ensemblepath + 'y_val_pred.npy')
    
    def evaluate(self, n_bootstrapping_samples=100, n_jobs=2, bootstrap_eval=False, dumped_bootstraps=True):
        # get labels
        y_train = np.load(self.outputfolder+self.experiment_name+'/data/y_train.npy', allow_pickle=True)
        y_test = np.load(self.outputfolder+self.experiment_name+'/data/y_test.npy', allow_pickle=True)
        
        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            if not dumped_bootstraps:
                test_samples = np.array(utils.get_appropriate_bootstrap(y_test, n_bootstrapping_samples))
            else:
                test_samples = np.load(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy', allow_pickle=True)
        else:
            test_samples = np.array([range(len(y_test))])
        
        # store samples for future evals
        test_samples.dump(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy')
        
        # iterate over all fitted models
        for m in sorted(os.listdir(self.outputfolder+self.experiment_name+'/models')):
            print(m)
            mpath = self.outputfolder+self.experiment_name+'/models/' + m + '/'
            rpath = self.outputfolder+self.experiment_name+'/models/' + m + '/results/'
            
            # load preds
            y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
            y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)
            
            pool = multiprocessing.Pool(n_jobs)
            
            thresholds = None # need to figure this one out
            
            te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds))))
            te_df_point = utils.generate_results(range(len(y_test)), y_test, y_test_pred, thresholds)
            te_df_result = pd.DataFrame(
                np.array([te_df_point.mean().values,
                         te_df.mean().values,
                         te_df.quantile(0.05).values,
                         te_df.quantile(0.95).values]), 
                columns=te_df.columns, 
                index=['point', 'mean', 'lower', 'upper'])
            
            pool.close()
            
            # dump results
            te_df_result.to_csv(rpath+'te_results.csv')