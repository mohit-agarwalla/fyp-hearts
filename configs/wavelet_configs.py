conf_wavelet_standard_lr = {
    'modelname': 'Wavelet+LR', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='LR')
}

conf_wavelet_standard_rf = {
    'modelname': 'Wavelet+RF', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_wavelet_standard_nn = {
    'modelname': 'Wavelet+NN', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet_single_rf = {
    'modelname': 'Wavelet+RF+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_wavelet_single_nn = {
    'modelname': 'Wavelet+NN+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet_standard_xgb = {
    'modelname': 'Wavelet+XGB', 
    'modeltype': 'WAVELET',
    'modelparams': dict(tree='hist', classifier='XGB')
}

conf_wavelet_single_xgb = {
    'modelname': 'Wavelet+XGB+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(tree='hist', classifier='XGB')
}