import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling1D, Convolution1D, BatchNormalization, Input, Activation, Add, Dropout, ReLU, LeakyReLU, Layer
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Input, Concatenate, Bidirectional, GRU
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs): 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform') 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer) 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint) 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint) 
            self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint) 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W) 
        if self.bias:
            uit += self.b 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u) 
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx()) 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

class GRUAttNet_Demo():
    def __init__(self, n_classes, freq, outputfolder, model_type=0, batchsize=32, 
                 epochs=100, dropout=0.1, num_blocks=5, final_act='lrelu'):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.model_type = model_type
        self.batchsize = batchsize
        self.epochs = epochs
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.final_act = final_act
    
    def build(self):
        inputs =Input(shape=(1000,1))
        conv1 = Convolution1D(12, 3, padding='same')(inputs)
        lrelu1 = LeakyReLU(alpha=0.3)(conv1)
        conv2 = Convolution1D(12, 3, padding='same')(lrelu1)
        lrelu2 = LeakyReLU(alpha=0.3)(conv2)
        conv3 = Convolution1D(12, 3, strides=2, padding='same')(lrelu2)
        lrelu3 = LeakyReLU(alpha=0.3)(conv3)
        x = Dropout(self.dropout)(lrelu3)
        
        for block in range(self.num_blocks-1):
            x = Convolution1D(12, 3, padding='same')(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = Convolution1D(12, 3, padding='same')(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = Convolution1D(12, 24, strides = 2, padding='same')(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = Dropout(self.dropout)(x)
        
        gru = Bidirectional(GRU(12, input_shape=(2250,12), return_sequences=True, return_state=False))(x)
        lrelu4 = LeakyReLU(alpha=0.3)(gru)
        dp4 = Dropout(self.dropout)(lrelu4)
        att = AttentionWithContext()(dp4)
        
        
        if self.model_type == 0:
            bn = BatchNormalization()(att)           
            dp5 = Dropout(0.1)(bn)
            if self.final_act=='lrelu':
                lrelu4 = LeakyReLU(alpha=0.3)(dp5)
            else:
                lrelu4 = ReLU()(dp5)
            
            dp5 = Dropout(0.1)(lrelu4)
            output = Dense(self.n_classes, activation='sigmoid')(dp5)
            model = Model(inputs=inputs, outputs=output)
            
        if self.model_type == 1:
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([att, demo_input])
            
            bn = BatchNormalization()(combined_layer) 
            dp5 = Dropout(0.1)(bn)
            if self.final_act=='lrelu':
                lrelu4 = LeakyReLU(alpha=0.3)(dp5)
            else:
                lrelu4 = ReLU()(dp5)
            
            dp5 = Dropout(0.1)(lrelu4)
            output = Dense(self.n_classes, activation='sigmoid')(dp5)
            model = Model(inputs=[inputs, demo_input], outputs=output)
        
        if self.model_type == 2:
            bn = BatchNormalization()(att)           
            dp5 = Dropout(0.1)(bn)
            if self.final_act=='lrelu':
                lrelu4 = LeakyReLU(alpha=0.3)(dp5)
            else:
                lrelu4 = ReLU()(dp5)
            
            dp5 = Dropout(0.1)(lrelu4)
            output = Dense(self.n_classes, activation='sigmoid')(dp5)
            
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([output, demo_input])
            output = Dense(self.n_classes, activation='sigmoid')(output)
            model = Model(inputs=[inputs, demo_input], outputs=output)
    
        return model
    
    def fit(self, X_train, y_train, demographic_train, X_val, y_val, demographic_val):
        model = self.build()
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
        model = load_model(self.outputfolder+'last_model.h5', custom_objects={"AttentionWithContext":AttentionWithContext})
        
        if type(demographic_test) == bool:
            return model.predict(X)
        else:
            return model.predict([X, demographic_test])
        
    @staticmethod
    def get_name():
        return 'gruattnet_demo'        
            

class AttNet_Demo():
    def __init__(self, n_classes, freq, outputfolder, model_type=0, batchsize=32, 
                 epochs=100, dropout=0.1, num_blocks=5, final_act='lrelu'):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.model_type = model_type
        self.batchsize = batchsize
        self.epochs = epochs
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.final_act = final_act
    
    def build(self):
        inputs =Input(shape=(1000,1))
        conv1 = Convolution1D(12, 3, padding='same')(inputs)
        lrelu1 = LeakyReLU(alpha=0.3)(conv1)
        conv2 = Convolution1D(12, 3, padding='same')(lrelu1)
        lrelu2 = LeakyReLU(alpha=0.3)(conv2)
        conv3 = Convolution1D(12, 3, strides=2, padding='same')(lrelu2)
        lrelu3 = LeakyReLU(alpha=0.3)(conv3)
        x = Dropout(self.dropout)(lrelu3)
        
        for block in range(self.num_blocks-1):
            x = Convolution1D(12, 3, padding='same')(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = Convolution1D(12, 3, padding='same')(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = Convolution1D(12, 24, strides = 2, padding='same')(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = Dropout(self.dropout)(x)
        
        # gru = Bidirectional(GRU(12, input_shape=(2250,12), return_sequences=True, return_state=False))(x)
        # lrelu4 = LeakyReLU(alpha=0.3)(x)
        # dp4 = Dropout(self.dropout)(lrelu4)
        att = AttentionWithContext()(x)
        
        
        if self.model_type == 0:
            bn = BatchNormalization()(att)           
            dp5 = Dropout(0.1)(bn)
            if self.final_act=='lrelu':
                lrelu4 = LeakyReLU(alpha=0.3)(dp5)
            else:
                lrelu4 = ReLU()(dp5)
            
            dp5 = Dropout(0.1)(lrelu4)
            output = Dense(self.n_classes, activation='sigmoid')(dp5)
            model = Model(inputs=inputs, outputs=output)
            
        if self.model_type == 1:
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([att, demo_input])
            
            bn = BatchNormalization()(combined_layer) 
            dp5 = Dropout(0.1)(bn)
            if self.final_act=='lrelu':
                lrelu4 = LeakyReLU(alpha=0.3)(dp5)
            else:
                lrelu4 = ReLU()(dp5)
            
            dp5 = Dropout(0.1)(lrelu4)
            output = Dense(self.n_classes, activation='sigmoid')(dp5)
            model = Model(inputs=[inputs, demo_input], outputs=output)
        
        if self.model_type == 2:
            bn = BatchNormalization()(att)           
            dp5 = Dropout(0.1)(bn)
            if self.final_act=='lrelu':
                lrelu4 = LeakyReLU(alpha=0.3)(dp5)
            else:
                lrelu4 = ReLU()(dp5)
            
            dp5 = Dropout(0.1)(lrelu4)
            output = Dense(self.n_classes, activation='sigmoid')(dp5)
            
            demo_input = Input(shape=(4,))
            combined_layer = Concatenate()([output, demo_input])
            output = Dense(self.n_classes, activation='sigmoid')(output)
            model = Model(inputs=[inputs, demo_input], outputs=output)
    
        return model
    
    def fit(self, X_train, y_train, demographic_train, X_val, y_val, demographic_val):
        model = self.build()
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
        model = load_model(self.outputfolder+'last_model.h5', custom_objects={'AttentionWithContext':AttentionWithContext})
        
        if type(demographic_test) == bool:
            return model.predict(X)
        else:
            return model.predict([X, demographic_test])
        
    @staticmethod
    def get_name():
        return 'attnet_demo'     