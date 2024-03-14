conf_wavelet_standard_lr = {
    'modelname': 'Wavelet+dLR', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='LR')
}

conf_wavelet_standard_rf = {
    'modelname': 'Wavelet+dRF', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_wavelet_standard_nn = {
    'modelname': 'Wavelet+dNN', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet_single_rf = {
    'modelname': 'Wavelet+dRF+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_wavelet_single_nn = {
    'modelname': 'Wavelet+dNN+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet_standard_xgb = {
    'modelname': 'Wavelet+dXGB', 
    'modeltype': 'WAVELET',
    'modelparams': dict(tree='hist', classifier='XGB')
}

conf_wavelet_single_xgb = {
    'modelname': 'Wavelet+dXGB+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(tree='hist', classifier='XGB')
}

# Demographic ones
conf_wavelet_demo_standard_lr = {
    'modelname': 'Wavelet+demo+LR', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='LR')
}

conf_wavelet_demo_standard_rf = {
    'modelname': 'Wavelet+demo+RF', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_wavelet_demo_standard_nn = {
    'modelname': 'Wavelet+demo+NN', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet_demo_single_rf = {
    'modelname': 'Wavelet+demo+RF+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_wavelet_demo_single_nn = {
    'modelname': 'Wavelet+demo+NN+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet_demo_standard_xgb = {
    'modelname': 'Wavelet+demo+XGB', 
    'modeltype': 'WAVELET',
    'modelparams': dict(tree='hist', classifier='XGB')
}

conf_wavelet_demo_single_xgb = {
    'modelname': 'Wavelet+demo+XGB+lead1', 
    'modeltype': 'WAVELET',
    'modelparams': dict(tree='hist', classifier='XGB')
}