conf_new_wavelet_standard_rf = {
    'modelname': 'NewWavelet+RF', 
    'modeltype': 'MODWAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_new_wavelet_standard_nn = {
    'modelname': 'NewWavelet+NN', 
    'modeltype': 'MODWAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_new_wavelet_500_standard_nn = {
    'modelname': 'NewWavelet+NN', 
    'modeltype': 'MODWAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN',freq=500)
}

conf_new_wavelet_single_rf = {
    'modelname': 'NewWavelet+RF+lead1', 
    'modeltype': 'MODWAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_new_wavelet_single_nn = {
    'modelname': 'NewWavelet+NN+lead1', 
    'modeltype': 'MODWAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_new_wavelet_standard_xgb = {
    'modelname': 'NewWavelet+XGB', 
    'modeltype': 'MODWAVELET',
    'modelparams': dict(tree='hist', classifier='XGB')
}

conf_new_wavelet_single_xgb = {
    'modelname': 'NewWavelet+XGB+lead1', 
    'modeltype': 'MODWAVELET',
    'modelparams': dict(tree='hist', classifier='XGB')
}
