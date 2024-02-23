import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dropout, Activation, BatchNormalization, Add, Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
import os       
    

class LSTM_Model():
    def __init__(self, filters=(64,64,32), dropout=0.3, **kwargs):
        self.filters = filters
        self.dropout = dropout
    
    def build(self, input_shape):
        return
        
class LSTM_Model():
    def __init__(self, n_classes, freq, outputfolder, leads=12, filters=(64,64,32,27), init_act='relu', end_act='sigmoid', batch_size=128):
        self.filters = filters
        self.init_act = init_act
        self.end_act = end_act
        self.name = 'lstm'
        self.n_classes = n_classes
        self.freq = freq
        self.epochs = 30
        self.batch_size = batch_size
        self.outputfolder = outputfolder + os.sep
        self.leads = leads
    
    def build(self):
        lstm_model = Sequential()
        lstm_model.add(LSTM(self.filters[0], input_shape=(10*self.freq,self.leads), return_sequences=True))
        lstm_model.add(LSTM(self.filters[1]))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(Dense(self.filters[2], activation = self.init_act))
        lstm_model.add(Dense(self.n_classes, activation = self.end_act))
        self.model =lstm_model
        return lstm_model

        
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.model = self.build()
        self.model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['acc', AUC(multi_label=True)])
        
        mc_score = ModelCheckpoint(self.outputfolder +'best_score_model.h5', monitor='val_keras_macro_auroc', mode='max', verbose=1, save_best_only=True)
        mc_loss = ModelCheckpoint(self.outputfolder+'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        self.model.fit(X_train,y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size)
        self.model.save(self.outputfolder+'last_model.h5')
    
    def predict(self, X):
        model = load_model(self.outputfolder+'best_loss_model.h5')
        return model.predict(X)
    
    @staticmethod
    def get_name():
        return 'lstm'