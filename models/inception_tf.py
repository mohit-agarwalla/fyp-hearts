import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Concatenate

def noop(x): return x

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding="same"):
    return Conv1D(in_planes, out_planes, kernel_size, strides=stride, padding=padding, use_bias=False)

def bn_drop_lin(n_in, n_out, bn=True, p=0.0, actn=None):
    model_layers = []
    if bn: model_layers.append(keras.layers.BatchNormalization(input_shape=(n_in,)))
    if p != 0: model_layers.append(keras.layers.Dropout(p))
    
    model_layers.append(keras.layers.Dense(n_out, activation=None))
    
    if actn is not None:
        model_layers.append(keras.layers.Activation(actn=actn))
    
    return model_layers

def create_head1d(nf, nc, lin_ftrs=None, ps=0.5, bn_final=False, bn=True, act='relu', concat_pooling=True):
    if lin_ftrs is None:
        lin_ftrs = [2*nf if concat_pooling else nf, nc]
    else:
        lin_ftrs = [2*nf if concat_pooling else nf] + lin_ftrs + [nc]
    
    ps = [ps] if isinstance(ps, float) else ps
    
    model_layers = []
    
    if concat_pooling:
        model_layers.append(keras.layers.GlobalAveragePooling1D())
    else:
        model_layers.append(keras.layers.MaxPooling1D(pool_size=2))
    
    model_layers.append(keras.layers.Flatten())
    
    for i in range(len(lin_ftrs)-1):
        ni, no = lin_ftrs[i], lin_ftrs[i+1]
        p = ps[min(i, len(ps)-1)]
        actn = act if i < len(lin_ftrs)-2 else None
        model_layers += bn_drop_lin(ni, no, bn, p, actn)
    
    if bn_final:
        model_layers.append(keras.layers.BatchNormalization())
    
    return keras.models.Sequential(model_layers)


class InceptionBlock1D(Layer):
    def __init__(self, ni, nb_filters, kss, stride=1, act='linear', bottleneck_size=32, **kwargs):
        super(InceptionBlock1D, self).__init__(**kwargs)
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size>0) else noop
        
        self.convs = [conv(bottleneck_size if bottleneck_size > 0 else ni, nb_filters, ks) for ks in kss]
        self.conv_bottle = tf.keras.Sequential([
            MaxPooling1D(3, strides=stride, padding='same'),
            conv(ni, nb_filters, 1)
        ])
        self.bn_relu = tf.keras.Sequential([
            BatchNormalization(),
            ReLU()
        ])
    
    def call(self, x, **kwargs):
        bottled = self.bottleneck(x) if self.bottleneck else noop(x)
        
        convs_outputs = self.bn_relu()
        
        convs_outputs = [conv(bottled) for conv in self.convs]
        convs_outputs.append(self.conv_bottle(x))
        concatenated = Concatenate(axis=-1)(convs_outputs)
        out = self.bn_relu(concatenated)
        return out

class Shortcut1D(Layer):
    def __init__(self, ni, nf, **kwargs):
        super(Shortcut1D, self).__init__(**kwargs)
        self.act_fn = ReLU(True)
        self.conv = conv(ni, nf, 1)
        self.bn = BatchNormalization()
    
    def call(self, inputs, residual):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act_fn(x+residual)
        return x

class InceptionBackBone(Layer): #GPT thought it should be Model
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual, **kwargs):
        super(InceptionBackBone, self).__init__()
        assert depth % 3 ==0
        
        self.use_residual = use_residual
        self.depth = depth
        n_ks = len(kss) + 1
        
        self.im = [InceptionBlock1D(input_channels if d == 0 else n_ks * nb_filters, nb_filters, kss, bottleneck_size=bottleneck_size) for d in range(depth)]
        self.sk = [Shortcut1D(input_channels if d == 0 else n_ks * nb_filters, n_ks * nb_filters) for d in range(depth // 3)] 

    def call(self, inputs, **kwargs):
        x = inputs
        for dep in range(self.depth):
            if self.use_residual and dep%3 == 2:
                residual = x
                
            x = self.im[dep][x]
            
            if self.use_residual and dep%3==2:
                x = self.sk[dep//3](residual, x)  
        return x

class Inception1D(keras.Model):
    def __init__(self, num_classes=2, input_channels=8, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32, use_residual=True, lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True,act_head="relu", concat_pooling=True, **kwargs):
        super(Inception1D, self).__init__()
        assert(kernel_size>40)
        kernel_size = [k-1 if k%2==0 else k for k in [kernel_size, kernel_size//2, kernel_size//4]]
        
        self.inception_backbone = InceptionBackBone(input_channels=input_channels, kss=kernel_size, depth=depth, bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual)
        
        n_ks = len(kernel_size) - 1
        
        # create head
        self.head = create_head1d(n_ks * nb_filters, num_classes=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        
    def call(self, inputs):
        x = self.inception_backbone(inputs)
        x = self.head(x)
        return x
    
    def get_output_layer(self):
        return self.head.layers[-1]
    
    def set_output_layer(self, new_layer):
        self.head.layers[-1] = new_layer
    
    def get_layer_groups(self):
        depth = len(self.inception_backbone.layers)
        if depth > 3:
            return (self.inception_backbone.layers[3:], self.head)
        else:
            return self.head 