from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, BatchNormalization, GlobalAveragePooling1D, ReLU, Dense, Flatten, Dropout, Layer
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
import os

class simple_block(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, dropout=0.2, **kwargs):
        super(simple_block, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout = dropout
    
    def build(self, input_shape):
        num_chan = input_shape[-1]
        self.conv1d = Conv1D(self.filters, self.kernel_size, strides=self.strides, dropout=self.dropout, padding='same', use_bias=False, kernel_initializer=VarianceScaling)
        self.av_pool = AveragePooling1D(pool_size=3, strides=self.strides, padding='valid')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.dropout = Dropout(self.dropout)
        
        super().build(input_shape)
    
    def call(self, x, **kwargs):
        x = self.conv1d(x)
        x = self.av_pool(x)
        x = self.relu(x)
        x = self.bn(x)
        
        x = self.dropout(x)
        return x

class simple_cnn(Layer):
    def __init__(self, dropout=0.2, filters=(32,64,128,256), block_fn=simple_block, **kwargs):
        super(simple_cnn, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.filters = filters
        self.loss = 'categorical_crossentropy'
        self.model_name = 'simple_cnn'
        self.block_fn = block_fn
            
    def build(self, input_shape):
        self.av_pool = AveragePooling1D(pool_size=3, strides=3, padding='valid')
        self.relu = ReLU()
        self.bn = BatchNormalization()
        self.blocks = []
        
        for filter in self.filters:
            block = self.block_fn(filters=filter, dropout=self.dropout)
            self.blocks.append(block)
        
        self.global_pool = GlobalAveragePooling1D()
        self.blocks.append(self.global_pool)
        super().build(input_shape)
        
    def get_optimizer(self, lr):
        return Adam(learning_rate=lr)

    @staticmethod
    def get_name():
        return 'simple_cnn'

class SimpleCNN():
    def __init__(self, n_classes, freq, outputfolder, epochs=10, batch_size=10):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder + os.sep
        self.epochs = epochs
        self.batch_size = batch_size
        
    def build(self):
        ann_model = Sequential()
        ann_model.add(Dense(50, activation='relu', input_shape=(10*self.freq,12)))
        ann_model.add(Dense(50, activation='relu'))
        ann_model.add(Dense(50, activation='relu'))
        ann_model.add(Dense(50, activation='relu'))
        ann_model.add(GlobalAveragePooling1D())
        ann_model.add(Dense(self.n_classes, activation='sigmoid'))
        return ann_model

    def fit(self, X, y, X_val, y_val):
        model = self.build()
        model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.5),Recall(name='Recall'),Precision(name='Precision'), AUC(
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name="AUC",
        dtype=None,
        thresholds=None,
        multi_label=True,
        label_weights=None)])
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        self.model = model
        self.model.save(self.outputfolder+"best_model.h5")
        return model
    
    def predict(self, X):
        model = load_model(self.outputfolder+'best_model.h5')
        
        preds = model.predict(X)
        return preds