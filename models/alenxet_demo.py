import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv1D, BatchNormalization, MaxPool1D
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from utils import CustomMetric as CM
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Input, Concatenate

class AlexNetDemo:
    def __init__(self, n_classes,freq, outputfolder, model_type=0, batchsize=128, epochs=100):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.model_type = model_type
        self.batchsize = batchsize
        self.epochs = epochs
    
    def fit(self, X_train, y_train, demographic_train, X_val, y_val, demographic_val):
        if self.model_type == 0:
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
        
        elif self.model_type == 1:
            inputs = Input(shape=(1000,1))
            conv1 = Conv1D(filters=96, kernel_size=11, strides=4, input_shape=(1000,1))(inputs)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation('relu')(bn1)
            mp1 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu1)
            conv2 = Conv1D(filters=256, kernel_size=5)(mp1)
            bn2 = BatchNormalization()(conv2)
            relu2 = Activation('relu')(bn2)
            mp2 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu2)
            conv3 = Conv1D(filters=384, padding='same', kernel_size=3)(mp2)
            bn3 = BatchNormalization()(conv3)
            relu3 = Activation('relu')(bn3)
            conv4 = Conv1D(filters=384, padding='same', kernel_size=3)(relu3)
            bn4 = BatchNormalization()(conv4)
            relu4 = Activation('relu')(bn4)
            conv5 =  Conv1D(filters=256, kernel_size=3)(relu4)
            bn5 = BatchNormalization()(conv5)
            relu5 = Activation('relu')(bn5)
            mp2 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu5)
            gp1 = GlobalAveragePooling1D()(mp2)
            
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([gp1, demo_input])
            
            dense1 = Dense(128, activation='relu')(combined_layer)
            dp1 = Dropout(0.4)(dense1)
            dense2 = Dense(128, activation='relu')(dp1)
            dp2 = Dropout(0.4)(dense2)
            final_output = Dense(self.n_classes, activation='sigmoid')(dp2)
            
            model = Model(inputs=[inputs, demo_input], outputs=[final_output])
        
        else:
            inputs = Input(shape=(1000,1))
            conv1 = Conv1D(filters=96, kernel_size=11, strides=4, input_shape=(1000,1))(inputs)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation('relu')(bn1)
            mp1 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu1)
            conv2 = Conv1D(filters=256, kernel_size=5)(mp1)
            bn2 = BatchNormalization()(conv2)
            relu2 = Activation('relu')(bn2)
            mp2 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu2)
            conv3 = Conv1D(filters=384, padding='same', kernel_size=3)(mp2)
            bn3 = BatchNormalization()(conv3)
            relu3 = Activation('relu')(bn3)
            conv4 = Conv1D(filters=384, padding='same', kernel_size=3)(relu3)
            bn4 = BatchNormalization()(conv4)
            relu4 = Activation('relu')(bn4)
            conv5 =  Conv1D(filters=256, kernel_size=3)(relu4)
            bn5 = BatchNormalization()(conv5)
            relu5 = Activation('relu')(bn5)
            mp2 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu5)
            gp1 = GlobalAveragePooling1D()(mp2)
            dense1 = Dense(128, activation='relu')(gp1)
            dp1 = Dropout(0.4)(dense1)
            dense2 = Dense(128, activation='relu')(dp1)
            dp2 = Dropout(0.4)(dense2)
            final_output = Dense(self.n_classes, activation='sigmoid')(dp2)   
            
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([final_output, demo_input])          
            outputs = Dense(self.n_classes, activation=self.final_activation)(combined_layer)
            
            model = Model(inputs=[inputs, demo_input], outputs=[outputs])
        
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                           tf.keras.metrics.Recall(name='Recall'),
                           tf.keras.metrics.Precision(name='Precision'),
                           tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", name="AUC",
                                                multi_label=True, label_weights=None)])
        mc_cm = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_cm_model.h5', monitor='val_CM', mode='max', verbose=1, save_best_only=True)
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        if self.model_type == 0:
            model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs, validation_data=(X_val, y_val), callbacks=[mc_cm, mc_loss])
        if self.model_type==1 or self.model_type==2:
            model.fit([X_train, demographic_train],y_train, validation_data=([X_val, demographic_val], y_val), callbacks=[mc_loss, mc_cm], epochs=self.epochs, batch_size=self.batchsize)
        
        model.save(self.outputfolder+'last_model.h5')
    
    def predict(self, X, demographic_test=False):
        model = load_model(self.outputfolder+'last_model.h5')
        
        if type(demographic_test) == bool:
            return model.predict(X)
        else:
            return model.predict([X, demographic_test])
    
    @staticmethod
    def get_name():
        return 'demo_alexnet'