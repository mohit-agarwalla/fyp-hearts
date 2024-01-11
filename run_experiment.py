from experiments.ecg_experiment import ECG_Experiment
import utils
import os
from configs.wavelet_configs import *

def main():
    
    outputfolder = os.getcwd()+'/output/'
    datafolder = os.getcwd()+'/datasets/PTB-XL/'
    
    models = [
        conf_wavelet_standard_rf
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
    experiments = [experiments[-1]]
    
    for name, task in experiments:
        e = ECG_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        e.perform()
        e.evaluate()
    
    # generate summary table
    utils.generate_summary_table()

if __name__ == '__main__':
    main()