conf_wavelet_lr = {
    'modelname': 'Wavelet+LR', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='LR')
}

conf_wavelet_demo_lr = {
    'modelname': 'Wavelet+demo+LR', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='LR')
}

conf_wavelet_rf = {
    'modelname': 'Wavelet+RF', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_wavelet_demo_rf = {
    'modelname': 'Wavelet+demo+RF', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='RF')
}

conf_wavelet_nn = {
    'modelname': 'NewWavelet+NN', 
    'modeltype': 'NewWavelet',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_newwavelet2_demo_nn = {
    'modelname': 'NewWavelet+demo+2+NN', 
    'modeltype': 'HCF2',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet2_demo_nn = {
    'modelname': 'Wavelet+demo+2+NN', 
    'modeltype': 'WAVELET2',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_newwavelet_demo_nn = {
    'modelname': 'NewWavelet+demo+NN', 
    'modeltype': 'NewWavelet',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_newwavelet_nn = {
    'modelname': 'Wavelet+NN', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet_demo_nn = {
    'modelname': 'Wavelet+demo+NN', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='NN')
}

conf_wavelet_xgb = {
    'modelname': 'Wavelet+XGB', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='XGB')
}

conf_wavelet_demo_xgb = {
    'modelname': 'Wavelet+demo+XGB', 
    'modeltype': 'WAVELET',
    'modelparams': dict(regularizer_C=0.001, classifier='XGB')
}