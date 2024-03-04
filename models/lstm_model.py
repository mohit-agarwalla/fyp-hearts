import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
import pickle

class Basic_LSTM():
    def __init__(self, n_classes, freq, outputfolder, batchsize=10, epochs=100, dropout=0.3, filters=[64,64,32], leads=12, activation='relu'):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.batch_size = batchsize
        self.epochs = epochs
        self.dropout = dropout
        self.filters = filters
        self.leads = leads
        self.activation = activation
        self.name = 'lstm'
    
    def fit(self, X_train, y_train, X_val, y_val):
        lstm = Sequential()
        lstm.add(LSTM(self.filters[0], input_shape=(1000,self.leads), return_sequences=True))
        lstm.add(LSTM(self.filters[1]))
        lstm.add(Dense(self.filters[2], activation=self.activation))
        lstm.add(Dropout(self.dropout))
        lstm.add(Dense(self.n_classes, activation='sigmoid'))
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', patience=3, verbose=1, min_delta=0, mode='max')
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        logger = CSVLogger(self.outputfolder+'training.log')
        
        lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(),
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy',threshold=0.5),
                      tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(num_thresholds=200,
                                           curve="ROC",
                                           summation_method='interpolation',
                                           name="AUC",
                                           multi_label=True,
                                           label_weights=None)])
        
        lstm.fit(X_train, y_train, batch_size=self.batch_size, 
                 epochs=self.epochs, validation_data=(X_val, y_val), 
                 callbacks=[early_stopping, mc_loss, logger])
        
        lstm.save(self.outputfolder+'last_model.h5')
        self.model = lstm
    
    def predict(self, X):
        model = load_model(self.outputfolder+'last_model.h5')
        return model.predict(X)
    
    @staticmethod
    def get_name():
        return 'lstm'