import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv1D, BatchNormalization, MaxPool1D, Activation, Add
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Input, Concatenate

class ResNetDemo():
    def __init__(self, n_classes, freq, outputfolder, model_type=0, batchsize=32, epochs=75, leads=1, filters=[64,64,64], **params):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.model_type = model_type
        self.batchsize = batchsize
        self.epochs = epochs
        self.filters = filters
        self.leads = leads
    
    def residual_block(self, x, filters, kernel_size=3,stride=1):
        y = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
        y = BatchNormalization()(y)

        if stride > 1:
            x = Conv1D(filters, kernel_size=1, strides=stride, padding='same')(x)

        out = Add()([x, y])
        out = Activation('relu')(out)
        return out
    
    def fit(self, X_train, y_train, demographic_train, X_val, y_val, demographic_val):
        input_layer =Input(shape=(1000,1))
        x = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        for i in range(len(self.filters)):
            x = self.residual_block(x, filters=self.filters[i])
        
        x = GlobalAveragePooling1D()(x)
        
        if self.model_type == 0:
            output_layer = Dense(self.n_classes, activation='softmax')(x)
            model = Model(inputs=input_layer, outputs=output_layer)
        elif self.model_type == 1:
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([x, demo_input])
            output_layer = Dense(self.n_classes, activation='softmax')(combined_layer)
            model = Model(inputs=[input_layer, demo_input], outputs=output_layer)
        elif self.model_type == 2:
            x = Dense(self.n_classes, activation='softmax')(x)
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([x, demo_input])
            output_layer = Dense(self.n_classes, activation='softmax')(combined_layer)
            model = Model(inputs=[input_layer, demo_input], outputs=output_layer)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_AUC', patience=3, verbose=1, min_delta=0, mode='max')
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        logger = tf.keras.callbacks.CSVLogger(self.outputfolder+'training.log')
        
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy',threshold=0.5),
                                tf.keras.metrics.Recall(name='recall'),
                                tf.keras.metrics.AUC(num_thresholds=200,
                                                    curve="ROC",
                                                    summation_method='interpolation',
                                                    name="AUC",
                                                    multi_label=True,
                                                    label_weights=None)])
        
        if self.model_type==0:
            model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs, 
                  validation_data=(X_val, y_val), callbacks=[early_stopping, mc_loss, logger])
        elif self.model_type==1 or self.model_type==2:
            model.fit([X_train, demographic_train], y_train, batch_size=self.batchsize, 
                      epochs=self.epochs, validation_data=[X_val, demographic_val], 
                      callbacks=[early_stopping, mc_loss, logger])
        
        model.save(self.outputfolder+'last_model.h5')
    
    def predict(self, X, demographic_test=False):
        model = load_model(self.outputfolder+'last_model.h5')
        
        if type(demographic_test) == bool:
            return model.predict(X)
        else:
            return model.predict([X, demographic_test])
        
    @staticmethod
    def get_name():
        return 'resnet_demo'  