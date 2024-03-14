import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from utils import CustomMetric as CM


class BaselineANN():
    def __init__(self, n_classes, freq, outputfolder, batchsize=10, epochs=100, leads=12, activation='relu', beta=0.5):
        # DISCLAIMER: Model assumes equal shapes across all samples
        
        # standard params
        self.name = 'ann'
        self.outputfolder = outputfolder + os.sep
        self.n_classes = n_classes
        self.freq = freq
        self.final_activation = 'sigmoid'
        self.activation = activation
        self.epochs = epochs
        self.batchsize = batchsize
        self.leads = leads
        self.beta = beta
    
    def fit(self, X_train, y_train, X_val,y_val):
        ann_model = Sequential()
        ann_model.add(Dense(50, activation=self.activation, input_shape=(1000,self.leads)))
        ann_model.add(Dense(50, activation=self.activation))
        ann_model.add(Dense(50, activation=self.activation))
        ann_model.add(Dense(50, activation=self.activation))
        ann_model.add(GlobalAveragePooling1D())
        ann_model.add(Dense(self.n_classes, activation=self.final_activation))
        
        ann_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                           tf.keras.metrics.Recall(name='Recall'),
                           tf.keras.metrics.Precision(name='Precision'),
                           tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", name="AUC",
                                                multi_label=True, label_weights=None),
                           CustomMetric(beta=self.beta, name='CM')])
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', patience=3, verbose=1, min_delta=0, mode='max')
        
        mc_cm = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_cm_model.h5', monitor='val_CM', mode='max', verbose=1, save_best_only=True)
        mc_aum = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_AUC.h5', monitor='val_AUC', mode='max', verbose=1, save_best_only=True)
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        ann_model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs, validation_data=(X_val, y_val), callbacks=[mc_cm, mc_loss])
        ann_model.save(self.outputfolder+'last_model.h5')
        self.model = ann_model
        
       
    def predict(self, X):
        model = load_model(self.outputfolder+'last_model.h5')
        return model.predict(X)
        
    
    @staticmethod
    def get_name():
        return 'simple_ann'