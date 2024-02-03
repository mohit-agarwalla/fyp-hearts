from experiments.demographic_experiment import DEMO_Experiment
import utils
import os
from configs.demo_wavelet_configs import *

def main():
    
    outputfolder = os.getcwd()+'/output/'
    datafolder = os.getcwd()+'/datasets/PTB-XL/'
    
    models = [
        conf_wavelet_standard_rf,
        conf_wavelet_demo_standard_rf,
#         conf_wavelet_standard_nn,
        # conf_wavelet_single_nn,
        conf_wavelet_single_rf,
        conf_wavelet_demo_single_rf,
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
    

        
    for name, task in experiments:
        f = DEMO_Experiment(name, task, datafolder, outputfolder, models)
        f.prepare()
        f.perform()
        f.evaluate()
    
    
    # generate summary table
    utils.generate_summary_table()

if __name__ == '__main__':
    main()