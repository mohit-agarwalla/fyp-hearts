import tensorflow as tf
from tensorflow.keras.metrics import Metric
from utils import *

class metric_func(Metric):
    def __init__(self, func, name='metric_func', ignore_idx=None, one_hot_encode_target=True, 
                 argmax_pred=False, softmax_pred=True, flatten_target=True, sigmoid_pred=False, 
                 metric_component=None, **kwargs):
        super(metric_func, self).__init__(name=name, **kwargs)
        self.func = func
        self.ignore_idx = ignore_idx
        self.one_hot_encode_target = one_hot_encode_target
        self.argmax_pred = argmax_pred
        self.softmax_pred = softmax_pred
        self.flatten_target = flatten_target
        self.sigmoid_pred = sigmoid_pred
        self.metric_component = metric_component
        self.name = name
    
    def reset_states(self):
        self.y_pred, self.y_true = [], []
    
    def result(self):
        y_pred = tf.concat(self.y_pred_list, axis=0)
        y_true = tf.concat(self.y_true_list, axis=0)
        metric_value = self.func(y_true, y_pred)
        
        if self.metric_component is not None:
            return metric_value[self.metric_component]
        else:
            return metric_value
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.flatten_target:
            y_true = tf.reshape(y_true, [-1])
        
        if self.argmax_pred:
            y_pred = tf.argmax(y_pred, axis=1)
        elif self.softmax_pred:
            y_pred = tf.nn.softmax(y_pred, axis=1)
        elif self.sigmoid_pred:
            y_pred = tf.sigmoid(y_pred)

        if self.ignore_idx is not None:
            mask = tf.not_equal(y_true, self.ignore_idx)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true - tf.boolean_mask(y_true, mask)
        
        if self.one_hot_encode_target:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        
        self.y_pred_list.append(y_pred)
        self.y_true_list.append(y_true)

def fmax_metric(targs,preds):
    return evaluate_experiment(targs,preds)['Fmax']

def auc_metric(targs,preds):
    return evaluate_experiment(targs,preds)["macro_auc"]

def mse_flat(preds, targs):
    return tf.reduce_mean(tf.square(tf.reshape(preds, [-1]) - tf.reshape(targs, [-1])))

def nll_regression(preds, targs):
    preds_mean = preds[:,0]
    preds_var = tf.clip_by_value(tf.exp(preds[:,1], 1e-4, 1e10))
    return tf.reduce_mean(0.5 * tf.math.log(2*np.pi*preds_var)) + tf.reduce_mean(tf.square(preds_mean - tf.squeeze(targs)) / (2 * preds_var))


