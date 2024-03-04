import tensorflow as tf
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPool1D, Input, Dense, GlobalAveragePooling1D
from keras.layers import concatenate
from keras.models import Model


class Inception():
    def __init__(self, n_classes, freq, outputfolder, batchsize=10, 
                 epochs=30, leads=12, blocks=5, filters=[64, 64, 64],
                 activation='relu', final_activation='sigmoid',
                 inception_filters=64):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.batchsize = batchsize
        self.epochs = epochs
        self.leads = leads
        self.blocks = blocks
        self.filters = filters
        self.activation = activation
        self.final_activation = final_activation
        self.inception_filters = inception_filters
    
    def inception_block(self, prev_layer):
        # conv1 
        conv1 = Conv1D(filters=self.inception_filters, kernel_size=1, padding='same')(prev_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        
        #conv2
        conv2 = Conv1D(filters=self.inception_filters, kernel_size=1, padding='same')(prev_layer)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv1D(filters=self.inception_filters, kernel_size=1, padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        
        #conv3
        conv3 = Conv1D(filters=self.inception_filters, kernel_size=1, padding='same')(prev_layer)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv1D(filters=self.inception_filters, kernel_size=1, padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        
        pool = MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
        convmax = Conv1D(filters=self.inception_filters, kernel_size=1, padding='same')(pool)
        convmax = BatchNormalization()(convmax)
        convmax = Activation('relu')(convmax)
        
        output = concatenate([conv1, conv2, conv3, convmax], axis=1)
        
        return output
        
        
    
    def get_model(self, input_shape):
        X_input = Input(input_shape)
        X = Conv1D(filters= self.filters[0], kernel_size=1, padding='same')(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        
        for block in range(self.blocks):
            X = self.inception_block(X)
            X = MaxPool1D(pool_size=2, strides=4, padding='same')(X)
        
        # Final part
        X = GlobalAveragePooling1D()(X)
        X = Dense(self.filters[1],activation=self.activation)(X)
        X = Dense(self.filters[2],activation=self.activation)(X)
        X = Dense(self.n_classes,activation=self.final_activation)(X)
        
        model = Model(inputs=X_input, outputs=X, name='Inception')
        
        return model
    
    def fit(self, X_train, y_train, X_val, y_val):
        model = self.get_model(input_shape=(1000, self.leads))
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy',threshold=0.5),
                      tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(num_thresholds=200,
                                           curve="ROC",
                                           summation_method='interpolation',
                                           name="AUC",
                                           multi_label=True,
                                           label_weights=None)])
        # callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', patience=3, verbose=1, min_delta=0, mode='max')
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        logger = tf.keras.callbacks.CSVLogger(self.outputfolder+'training.log')
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, callbacks=[early_stopping, mc_loss, logger])

        model.save(self.outputfolder+'last_model.h5')
        self.model = model
    
    def predict(self, X):
        model = load_model(self.outputfolder+'last_model.h5')
        return model.predict(X)

    @staticmethod
    def get_name():
        return 'inception'
        
    
    

    
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
                                                multi_label=True, label_weights=None)])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', patience=3, verbose=1, min_delta=0, mode='max')
        
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        ann_model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs, validation_data=(X_val, y_val), callbacks=[early_stopping, mc_loss])
        ann_model.save(self.outputfolder+'last_model.h5')
        self.model = ann_model
        
       
    def predict(self, X):
        model = load_model(self.outputfolder+'last_model.h5')
        return model.predict(X)
        
    
    @staticmethod
    def get_name():
        return 'simple_ann'