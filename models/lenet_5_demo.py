import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv1D, BatchNormalization, MaxPool1D, Activation
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Input, Concatenate

class LeNet_5_Demo:
    def __init__(self, n_classes, freq, outputfolder, model_type=0, batchsize=10, epochs=100, leads=1, activation='relu'):
        self.name = 'alexnet'
        self.outputfolder = outputfolder + os.sep
        self.model_type = model_type
        self.n_classes = n_classes
        self.freq = freq
        self.final_activation = 'sigmoid'
        self.activation = activation
        self.epochs = epochs
        self.batchsize = batchsize
        self.leads = leads
    
    def fit(self, X_train, y_train, demographic_train, X_val, y_val, demographic_val):
        if self.model_type == 0:
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
        elif self.model_type == 1:
            inputs = Input(shape=(1000,1))
            conv1 = Conv1D(filters=6, kernel_size=3, padding='same', input_shape=(1000,1))(inputs)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation('relu')(bn1)
            mp1 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu1)
            conv2 = Conv1D(filters=16, strides=1, kernel_size=5)(mp1)
            bn2 = BatchNormalization()(conv2)
            relu2 = Activation('relu')(bn2)
            mp2 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu2)
            gp1 = GlobalAveragePooling1D()(mp2)
            
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([gp1, demo_input])
            
            dense1 = Dense(64, activation='relu')(combined_layer)
            dense2 = Dense(32, activation='relu')(dense1)
            outputs = Dense(self.n_classes, activation = 'sigmoid')(dense2)
            
            lenet_5_model = Model(inputs=[inputs, demo_input], outputs=[outputs])
        elif self.model_type == 2:
            inputs = Input(shape=(1000,1))
            conv1 = Conv1D(filters=6, kernel_size=3, padding='same', input_shape=(1000,1))(inputs)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation('relu')(bn1)
            mp1 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu1)
            conv2 = Conv1D(filters=16, strides=1, kernel_size=5)(mp1)
            bn2 = BatchNormalization()(conv2)
            relu2 = Activation('relu')(bn2)
            mp2 = MaxPool1D(pool_size=2, strides=2, padding='same')(relu2)
            gp1 = GlobalAveragePooling1D()(mp2)
            dense1 = Dense(64, activation='relu')(gp1)
            dense2 = Dense(32, activation='relu')(dense1)
            outputs = Dense(self.n_classes, activation = 'sigmoid')(dense2)
            
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([outputs, demo_input])
            
            final_outputs = Dense(self.n_classes, activation = 'sigmoid')(combined_layer)
            
            lenet_5_model = Model(inputs=[inputs, demo_input], outputs=[final_outputs])
        
        lenet_5_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                           tf.keras.metrics.Recall(name='Recall'),
                           tf.keras.metrics.Precision(name='Precision'),
                           tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", name="AUC",
                                                multi_label=True, label_weights=None)])
        mc_cm = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_cm_model.h5', monitor='val_CM', mode='max', verbose=1, save_best_only=True)
        mc_loss = tf.keras.callbacks.ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        if self.model_type == 0:
            lenet_5_model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs, validation_data=(X_val, y_val), callbacks=[mc_cm, mc_loss])
        if self.model_type== 1 or self.model_type ==2:
            lenet_5_model.fit([X_train, demographic_train],y_train, validation_data=([X_val, demographic_val], y_val), callbacks=[mc_loss, mc_cm], epochs=self.epochs, batch_size=self.batchsize)
        
        lenet_5_model.save(self.outputfolder+'last_model.h5')
        
    def predict(self, X, demographic_test=False):
        model = load_model(self.outputfolder+'last_model.h5')
        
        if type(demographic_test) == bool:
            return model.predict(X)
        else:
            return model.predict([X, demographic_test])
    
    @staticmethod
    def get_name():
        return 'demo_lenet'