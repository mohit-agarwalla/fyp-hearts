import utils
import os
import pickle
import pandas as pd
import numpy as np
import mutilprocessing
from itertools import repeat
from models.baseline_wavelet import WaveletModel
from models.baseline_resnet import Resnet
from models.classifer import Classifier
# from models.baseline_conv1d import SimpleCNN
from models.baseline_ann import BaselineANN
# from models.lstm import LSTM_Model
from models.lstm_model import Basic_LSTM
from tensorflow.keras.callbacks import EarlyStopping
from models.baseline_resnet import Resnet
from models.cpsc_winner import AttentionWithContext
from models.lenet_5 import LeNet_5
from models.alexnet import AlexNet
from tensorflow.keras.models import load_model

class Ensembler:
    def __init__(self, experiment_name, task, datafolder, outputfolder, models, ensembler_name, sampling_frequency=100, min_samples=100, train_fold=8, val_fold=9, test_fold=10):
        self.experiment_name = experiment_name
        self.task = task
        self.datafolder = datafolder
        self.outputfolder = outputfolder
        self.models = models
        self.ensembler_name = ensembler_name
        self.sampling_frequency = sampling_frequency
        self.min_samples = min_samples
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        
    
    def complete(self, n_bootstrapping_samples=100, n_jobs=2, bootstrap_eval=False, dumped_bootstraps=True):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)
        
        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)
        
        # One hot encode relevant data
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')
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
        
        self.X_train = self.X_train[:,:,0].reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test[:,:,0].reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        self.X_val = self.X_val[:,:,0].reshape(self.X_val.shape[0], self.X_val.shape[1], 1)
        
        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            if not dumped_bootstraps:
                test_samples = np.array(utils.get_appropriate_bootstrap(self.y_test, n_bootstrapping_samples))
            else:
                test_samples = np.load(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy', allow_pickle=True)
        else:
            test_samples = np.array([range(len(self.y_test))])
        
        # store samples for future evals
        test_samples.dump(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy')
        res_df = pd.DataFrame(columns=['AUC', 'CustomMetric', 'val_AUC', 'val_CustomMetric'])
        preds_df = pd.DataFrame(['train', 'val', 'test'])
        
        for model in self.models:
            modelname = model['modelname']
            mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
            if os.path.exists(mpath+'best_cm.h5'):
                model = load_model(mpath+'best_cm.h5', custom_objects={'AttentionWithContext':AttentionWithContext, 'CustomMetric':utils.CustomMetric})
            elif os.path.exists(mpath+'best_loss.h5'):
                model = load_model(mpath+'best_loss.h5', custom_objects={'AttentionWithContext':AttentionWithContext, 'CustomMetric':utils.CustomMetric})
            else:
                model = load_model(mpath+'last_model.h5', custom_objects={'AttentionWithContext':AttentionWithContext, 'CustomMetric':utils.CustomMetric})
            
            test_preds = model.predict(self.X_test)
            train_preds = model.predict(self.X_train)
            val_preds = model.predict(self.X_val)
            
            preds_df.loc[modelname] = [train_preds, val_preds, test_preds]
            
            
            
            
            
            
            
            