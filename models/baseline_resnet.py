""""
Adapted from: ___
"""

from tensorflow.keras.layers import BatchNormalization, Dropout, Conv1D, Layer, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import ReLU as ReLu
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam

def conv1d(filters, kernel_size=3, strides=1):
    return Conv1D(filters, kernel_size, strides=strides, 
                  padding='same', use_bias=False, kernel_initializer=VarianceScaling)

class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout = dropout
        
    def build(self,input_shape):
        num_chan = input_shape[-1]
        self.conv1 = conv1d(self.filters, self.kernel_size, self.strides)
        self.bn1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = ReLu()
        self.dropout1 = Dropout(self.dropout)
        self.conv2 = conv1d(self.filters, self.kernel_size, 1)
        self.bn2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu2 = ReLu()

        if num_chan != self.filters or self.strides > 1:
            self.proj_conv = conv1d(self.filters, 1, self.strides)
            self.proj_bn = BatchNormalization(momentum=0.9, epsilon=1e-5)        
            self.projection = True
        else:
            self.projection = False
        
        super().build(input_shape)
    
    def call(self, x, **kwargs):
        shortcut = x
        
        if self.projection:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_bn(shortcut) 
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x + shortcut)       
        
        return x

class ResNet(Layer):
    def __init__(self, blocks=(2,2,2,2), filters=(64, 128, 256, 512),
                 kernel_size=(3, 3, 3, 3), block_fn=ResidualBlock, 
                 dropout=0.1, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.filters = filters
        self.block_nums = blocks
        self.kernel_size = kernel_size
        self.block_fn = block_fn
        self.dropout = dropout
        self.loss = 'categorial_crossentropy'
        self.model_name = 'resnet'
    
    def build(self, input_shape):
        self.conv1 = conv1d(64, 7, 2)
        self.bn1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = ReLu()
        self.maxpool1 = MaxPooling1D(3, 2, padding='same')
        self.blocks = []
        
        for stage, num_blocks in enumerate(self.block_nums):
            for block in range(num_blocks):
                strides = 2 if block == 0 and stage > 0 else 1
                res_block = self.block_fn(self.filters[stage], self.kernel_size[stage], strides, self.dropout)
                self.blocks.append(res_block)
        
        self.global_pool = GlobalAveragePooling1D()
        super().build(input_shape)
    
    def get_optimizer(self, lr):
        return Adam(lr=lr)
    
    
    @staticmethod
    def get_name():
        return 'resnet'        