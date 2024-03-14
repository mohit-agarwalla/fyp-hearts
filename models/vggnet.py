import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv1D, BatchNormalization, MaxPool1D, Activation, Dropout
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from utils import CustomMetric
from tensorflow.keras.callbacks import CSVLogger

class VGGNet:
    def __init__(self, n_classes, freq, outputfolder, batchsize=128, epochs=100, leads=1):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.batchsize = batchsize
        self.epochs = epochs
        self.leads = leads
    
    def fit(self, X_train, y_train, X_val, y_val):
        vgg_16_model=Sequential()

        vgg_16_model.add(Conv1D(filters=64, kernel_size=3, padding='same',  input_shape=(1000,self.leads)))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

        vgg_16_model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

        vgg_16_model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

        vgg_16_model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

        vgg_16_model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(Conv1D(filters=512, kernel_size=1, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(Conv1D(filters=512, kernel_size=1, padding='same'))
        vgg_16_model.add(BatchNormalization())
        vgg_16_model.add(Activation('relu'))
        vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

        vgg_16_model.add(GlobalAveragePooling1D())
        vgg_16_model.add(Dense(256, activation='relu'))
        vgg_16_model.add(Dropout(0.4))
        vgg_16_model.add(Dense(128, activation='relu'))
        vgg_16_model.add(Dropout(0.4))
        vgg_16_model.add(Dense(self.n_classes, activation='sigmoid'))
        
        vgg_16_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[tf.keras.metrics.AUC(num_thresholds=1000000, multi_label=False, name='AUC')]
        )
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', patience=3, verbose=1, min_delta=0, mode='max')
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        logger = CSVLogger(self.outputfolder+'training.log')
        
        vgg_16_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batchsize, callbacks=[es, mc_loss, logger])
        vgg_16_model.save(self.outputfolder+'last_model.h5')
    
    def predict(self, X):
        model = load_model(self.outputfolder+'best_loss.h5')
        return model.predict(X)
    
    @staticmethod
    def get_name():
        return 'vggnet'