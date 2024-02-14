conf_resnet_standard = {
    'modelname': 'ResNet+Baseline', 
    'modeltype': 'RESNET',
    'modelparams': dict(blocks=(2,2,2,2), filters=(64, 128, 256, 512), kernel_size=(3, 3, 3, 3))
}

conf_resnet_single = {
    'modelname': 'ResNet+Baseline+lead1', 
    'modeltype': 'RESNET',
    'modelparams': dict(blocks=(2,2,2,2), filters=(64, 128, 256, 512), kernel_size=(3, 3, 3, 3))
}