import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv1D, BatchNormalization, MaxPool1D, Activation, Dropout
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from utils import CustomMetric

class AlexNet():
    def __init__(self, n_classes,freq, outputfolder, batchsize=128, epochs=100, leads=1, beta=0.5):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.batchsize = batchsize
        self.beta = beta
        self.epochs = epochs
    
    def fit(self, X_train, y_train, X_val, y_val):
        # Create model
        model = Sequential()
        model.add(Conv1D(filters=96, kernel_size=11, strides=4, input_shape=(1000,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
        model.add(Conv1D(filters=256, kernel_size=5))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
        model.add(Conv1D(filters=384, padding='same', kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(filters=384, padding='same', kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(filters=256, kernel_size=3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.n_classes, activation='sigmoid'))
        
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[tf.keras.metrics.AUC(num_thresholds=1000000, multi_label=False),
                     CustomMetric(beta=self.beta)]
        )
        
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        mc_auc = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_auc.h5', monitor='val_AUC', mode='max', verbose=1, save_best_only=True)
        mc_cm = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_cm.h5', monitor='val_custom_metric', mode='max', verbose=1, save_best_only=True)
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batchsize, callbacks=[mc_auc, mc_loss, mc_cm])
        model.save(self.outputfolder+'last_model.h5')
        self.model = model
    
    def predict(self, X):
        model = load_model(self.outputfolder+'best_loss.h5', custom_objects={'CustomMetric':CustomMetric})
        return model.predict(X)

    @staticmethod
    def get_name():
        return 'alexnet'

