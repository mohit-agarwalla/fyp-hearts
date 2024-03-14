import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from utils import CustomMetric as CM
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Input, Concatenate

class ANNDemo():
    def __init__(self, n_classes, freq, outputfolder, model_type=0, batchsize=10, epochs=100, leads=12, activation='relu', beta=0.5):
        # standard params
        self.name = 'ann'
        self.outputfolder = outputfolder + os.sep
        self.model_type = model_type
        self.n_classes = n_classes
        self.freq = freq
        self.final_activation = 'sigmoid'
        self.activation = activation
        self.epochs = epochs
        self.batchsize = batchsize
        self.leads = leads
        self.beta = beta
    
    def fit(self, X_train, y_train, demographic_train, X_val, y_val, demographic_val):
        if self.model_type == 0:
            ann_model = Sequential()
            ann_model.add(Dense(50, activation=self.activation, input_shape=(1000,1)))
            ann_model.add(Dense(50, activation=self.activation))
            ann_model.add(Dense(50, activation=self.activation))
            ann_model.add(Dense(50, activation=self.activation))
            ann_model.add(GlobalAveragePooling1D())
            ann_model.add(Dense(self.n_classes, activation=self.final_activation))
        
        elif self.model_type == 1:
            inputs = Input(shape=(1000,1))
            dense1 = Dense(50, activation=self.activation)(inputs)
            dense2 = Dense(50, activation=self.activation)(dense1)
            dense3 = Dense(50, activation=self.activation)(dense2)
            dense4 = Dense(50, activation=self.activation)(dense3)
            gp1 = GlobalAveragePooling1D()(dense4)
            
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([gp1, demo_input])
            outputs = Dense(self.n_classes, activation=self.final_activation)(combined_layer)
            
            ann_model = Model(inputs=[inputs, demo_input], outputs=outputs)
        
        else:
            inputs = Input(shape=(1000,1))
            dense1 = Dense(50, activation=self.activation)(inputs)
            dense2 = Dense(50, activation=self.activation)(dense1)
            dense3 = Dense(50, activation=self.activation)(dense2)
            dense4 = Dense(50, activation=self.activation)(dense3)
            gp1 = GlobalAveragePooling1D()(dense4)
            final_dense = Dense(self.n_classes, activation=self.final_activation)(gp1)
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([final_dense, demo_input])
            outputs = Dense(self.n_classes, activation=self.final_activation)(combined_layer)
            
            ann_model = Model(inputs=[inputs, demo_input], outputs=outputs)
        
        ann_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                           tf.keras.metrics.Recall(name='Recall'),
                           tf.keras.metrics.Precision(name='Precision'),
                           tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", name="AUC",
                                                multi_label=True, label_weights=None)])
        mc_cm = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_cm_model.h5', monitor='val_CM', mode='max', verbose=1, save_best_only=True)
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
                        
        if self.model_type == 0:
            ann_model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs, validation_data=(X_val, y_val), callbacks=[mc_cm, mc_loss])
        if self.model_type==1 or self.model_type==2:
            ann_model.fit([X_train, demographic_train],y_train, validation_data=([X_val, demographic_val], y_val), callbacks=[mc_loss, mc_cm], epochs=self.epochs, batch_size=self.batchsize)
        
        ann_model.save(self.outputfolder+'last_model.h5')
    
    def predict(self, X, demographic_test=False):
        model = load_model(self.outputfolder+'last_model.h5')
        if type(demographic_test) == bool:
            return model.predict(X)
        else:
            return model.predict([X, demographic_test])
    
    @staticmethod
    def get_name():
        return 'demo_ann'