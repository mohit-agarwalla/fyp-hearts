import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import load_model

class Resnet():
    def __init__(self, n_classes, freq, outputfolder, batchsize=32, epochs=75, leads=12, filters=[64,64,64], **params):
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.batchsize = batchsize
        self.epochs = epochs
        self.leads = leads
        self.filters = filters
    
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
        
        
        
    def fit(self, X_train, y_train, X_val, y_val):
        input_layer = Input(shape=(1000, self.leads))
        x = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        for i in range(len(self.filters)):
            x = self.residual_block(x, filters=self.filters[i])
        
        # Global Average Pooling and Dense Layer
        x = GlobalAveragePooling1D()(x)
        output_layer = Dense(self.n_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        
        # Callbacks
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
        
        model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs, 
                  validation_data=(X_val, y_val), callbacks=[early_stopping, mc_loss, logger])
        
        model.save(self.outputfolder+"last_model.h5") 
        
    def predict(self, X):
        model = load_model(self.outputfolder+"last_model.h5")
        return model.predict(X)
    
    @staticmethod
    def get_name():
        return 'resnet'
         
            