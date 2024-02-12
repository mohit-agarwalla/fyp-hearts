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
from models.wavelet_demo import WaveletModel
from models.baseline_resnet import ResNet
from models.classifer import Classifier
from tensorflow.keras.callbacks import EarlyStopping


class DEMO_Experiment:
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
        subset=['age', 'weight', 'sex', 'height']
        # Get data only where demographic data is available
        self.labels, self.data = utils.select_subset(self.raw_labels, self.data, subset=subset)
        self.demographics = self.labels[subset]
        self.labels = utils.compute_label_aggregations(self.labels, self.datafolder, self.task)
        
        # One hot encode relevant data
        self.data, self.labels, self.Y, self.demographics, _ = utils.select_data_demo(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/', self.demographics)
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
        
        # collect demographic data 
        self.demographic_test = self.demographics[self.labels.strat_fold == self.test_fold]
        self.demographic_val = self.demographics[self.labels.strat_fold == self.val_fold]
        self.demographic_train = self.demographics[self.labels.strat_fold <= self.train_fold]
        
        # Save demographic data 
        np.array(self.demographic_train).dump(self.outputfolder + self.experiment_name+ '/data/demographic_train.npy')
        np.array(self.demographic_val).dump(self.outputfolder + self.experiment_name+ '/data/demographic_val.npy')
        np.array(self.demographic_test).dump(self.outputfolder + self.experiment_name+ '/data/demographic_test.npy') 

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
            
            if '+demo+' in modelname:
                demographic_train = self.demographic_train
                demographic_test = self.demographic_test
                demographic_val = self.demographic_val
            else:
                demographic_train, demographic_test, demographic_val = False, False, False
            
            # load respective model
            if modeltype == 'WAVELET':
                model = WaveletModel(n_classes, self.sampling_frequency, mpath, **modelparams)
                # fit model
                model.fit(X_train, self.y_train, demographic_train, X_val, self.y_val, demographic_val)
            if modeltype == 'RESNET':
                resnet = ResNet(**modelparams)
                model = Classifier(model=model, input_size=1000, learning_rate=0.0001)
                model.add_compile()
                es = EarlyStopping(monitor='val_loss', patience=6)
                model.fit(X_train, self.y_train, (X_val,self.y_val), mpath)
                

            # predict and dump
            model.predict(X_train, demographic_train).dump(mpath+'y_train_pred.npy')
            model.predict(X_val, demographic_val).dump(mpath+'y_val_pred.npy')
            model.predict(X_test, demographic_test).dump(mpath+'y_test_pred.npy')
            
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
        
        print(len(ensemble_test))
        print(f"ensemble test -1 is {ensemble_test[-1]}")
        print(f"Ensemble val is {(ensemble_val)}") 
        print(f"Ensemble test is {(ensemble_test)}") 
        print(f"{(ensemble_train)}") 
        print(f"Ensemble train shape is {np.array(ensemble_train).shape}") 
        
        print(f"Ensemble test shape is {np.array(ensemble_test).shape}") 
        
        print(f"Ensemble val shape is {np.array(ensemble_val).shape}")  
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
            
#             te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds))))

            result_list = []
            for test_sample, y_true, y_pred, threshold in zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds)):
                result = utils.generate_results(test_sample, y_true, y_pred, threshold)
                result_list.append(result)

            te_df = pd.concat(result_list)
            
            
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
            te_df_result.to_csv(rpath+'demo_te_results.csv')