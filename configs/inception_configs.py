# conf_inception1d = {'modelname':'inception1d', 'modeltype':'fastai_model', 
#     'modelparams':dict()}

# conf_inception1d_input256 = {'modelname':'inception1d_input256', 'modeltype':'fastai_model', 
#     'modelparams':dict(input_size=256)}

# conf_inception1d_input512 = {'modelname':'inception1d_input512', 'modeltype':'fastai_model', 
#     'modelparams':dict(input_size=512)}

# conf_inception1d_input1000 = {'modelname':'inception1d_input1000', 'modeltype':'fastai_model', 
#     'modelparams':dict(input_size=1000)}
    

# conf_inception1d_no_residual = {'modelname':'inception1d_no_residual', 'modeltype':'fastai_model', 
#     'modelparams':dict()}

conf_inceptionse  = {
    'modelname': 'InceptionSE+lead1', 
    'modeltype': 'InceptionSE',
    'modelparams': dict()
}

conf_inceptionse_64 = {
    'modelname': 'InceptionSE64+lead1', 
    'modeltype': 'InceptionSE',
    'modelparams': dict(filters=[64,64,64,64])
}

conf_inceptionse_128 = {
    'modelname': 'InceptionSE128+lead1', 
    'modeltype': 'InceptionSE',
    'modelparams': dict(filters=[128,128,128,128])
}