from experiments.ecg_experiment import ECG_Experiment
# from experiments.demographic_experiment import DEMO_Experiment
import utils
import os
from configs.wavelet_configs import *
from configs.inception_configs import *
from configs.resnet_configs import *

def main():
    
    outputfolder = os.getcwd()+'/output/'
    datafolder = os.getcwd()+'/datasets/PTB-XL/'
    
    models = [
        # conf_resnet_standard,
        conf_resnet_single,
        # conf_inception1d,
        # conf_inception1d_input256,
        # conf_inception1d_input512,
        # conf_inception1d_input1000,
        # conf_inception1d_no_residual,
        # conf_wavelet_standard_rf,
        # conf_wavelet_standard_nn,
        # conf_wavelet_single_nn,
        # conf_wavelet_single_rf,
        # conf_wavelet_single_xgb,
        # conf_wavelet_standard_xgb,
    ]
    
    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
    ]
    
    # for test (REMOVE BEFORE FULL RUN)
#     experiments = [experiments[-1]]
    
    # for name, task in experiments:
    #     e = ECG_Experiment(name, task, datafolder, outputfolder, models)
    #     e.prepare()
    #     e.perform()
    #     e.evaluate(bootstrap_eval=True, dumped_bootstraps=False)
        
    # for name, task in experiments:
    #     f = DEMO_Experiment(name, task, datafolder, outputfolder, models)
    #     f.prepare()
    #     f.perform()
    #     f.evaluate()
    
    
    # generate summary table
    utils.generate_summary_table()

if __name__ == '__main__':
    main()