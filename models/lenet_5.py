import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv1D, BatchNormalization, MaxPool1D, Activation
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from utils import CustomMetric

class LeNet_5():
    def __init__(self, n_classes,freq, outputfolder, batchsize=128, epochs=100, leads=1, beta=0.5):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.batchsize = batchsize
        self.beta = beta
        self.epochs = epochs
    
    def fit(self, X_train, y_train, X_val, y_val):
        # Create model
        lenet_5_model=Sequential()
        lenet_5_model.add(Conv1D(filters=6, kernel_size=3, padding='same', input_shape=(1000,1)))
        lenet_5_model.add(BatchNormalization())
        lenet_5_model.add(Activation('relu'))
        lenet_5_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
        lenet_5_model.add(Conv1D(filters=16, strides=1, kernel_size=5))
        lenet_5_model.add(BatchNormalization())
        lenet_5_model.add(Activation('relu'))
        lenet_5_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
        lenet_5_model.add(GlobalAveragePooling1D())
        lenet_5_model.add(Dense(64, activation='relu'))
        lenet_5_model.add(Dense(32, activation='relu'))
        lenet_5_model.add(Dense(self.n_classes, activation = 'sigmoid'))
        
        lenet_5_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[tf.keras.metrics.AUC(num_thresholds=1000000, multi_label=False),
                     CustomMetric(beta=self.beta)]
        )
        
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        mc_auc = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_auc.h5', monitor='val_AUC', mode='max', verbose=1, save_best_only=True)
        mc_cm = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_cm.h5', monitor='val_custom_metric', mode='max', verbose=1, save_best_only=True)
        
        lenet_5_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batchsize, callbacks=[mc_auc, mc_loss, mc_cm])
        lenet_5_model.save(self.outputfolder+'last_model.h5')
        self.model = lenet_5_model
    
    def predict(self, X):
        model = load_model(self.outputfolder+'best_loss.h5', custom_objects={'CustomMetric':CustomMetric})
        return model.predict(X)
    
    @staticmethod
    def get_name():
        return 'lenet_5'
              
        