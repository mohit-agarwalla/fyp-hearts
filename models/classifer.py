import numpy as np
from tensorflow.keras.metrics import AUC
# from wandb.keras import WandbCallback
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
# from metrics import TimeHistory
from utils import TimeHistory

class Classifier(Model):
    def __init__(self, model, input_size, n_classes, learning_rate=0.0001, epochs=20, path="output/", leads=12):
        super(Classifier, self).__init__()
        self.model = model
        self.input_size = input_size
        self.n_classes = n_classes
        self.lr = learning_rate
        self.epochs = epochs
        self.path = path
        self.leads = leads

        # Activation fn on end depends on number of outputs
        out_act = 'sigmoid' if n_classes > 1 else 'softmax'
        units = n_classes if n_classes>2 else 1
        self.classifer = Dense(units=units, input_shape=(leads*input_size,), activation=out_act)        
        
    def add_compile(self):
        self.compile(optimizer=self.model.get_optimizer(self.lr),loss = self.model.loss, metrics = ['acc', AUC(multi_label=True)])
        
    def summary(self):
        input_layer = Input(shape=(self.input_size, self.leads,), dtype='float32')
        model = Model(inputs=input_layer, outputs=self.call(input_layer))
        
        return model.summary()
    
    def call(self, x, **kwargs):
        print(x.shape)
        x = self.model(x)
        x = Flatten()(x)
        x = self.classifer(x)
        return x
    
    def fit(self, x, y, validation_data):
        time_callback = TimeHistory()
        
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#         wandb_cb = WandbCallback(save_weights_only=True)
        
        X_val,y_val = validation_data[0], validation_data[1]
        
        super(Classifier, self).fit(x, y, validation_data=(X_val,y_val), callbacks=[es, time_callback], epochs=self.epochs, batch_size=128) #callbacks=[es, time_callback, wandb_cb]
        times = time_callback.times
        return times