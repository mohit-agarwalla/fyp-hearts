from experiments.ecg_experiment import ECG_Experiment
# from experiments.demographic_experiment import DEMO_Experiment
import utils
import os
from configs.wavelet_configs import *
from configs.inception_configs import *
from configs.resnet_configs import *
from configs.cnn_configs import *
from configs.lstm_configs import *
from configs.baseline_ann_configs import *
from configs.new_wavelet_configs import *
from configs.lenet_configs import *
from configs.att_net import *
from configs.alexnet_configs import *
from configs.vggnet_configs import *


def main():
    
    outputfolder = os.getcwd()+'/output/'
    datafolder = os.getcwd()+'/datasets/PTB-XL/'
    
    models = [
        # conf_lenet,
        # conf_attnet_standard,
        # conf_attnet_dp02,
        # conf_alexnet,
        # conf_vggnet_lead1,
        # conf_wavelet_single_rf,
        # conf_lstm_lead1,
        # conf_lstm_attn,
        # conf_inceptionse,
        # conf_alexnetattn,
        # conf_lenet_attn,
        # conf_ann_standard,
        # conf_ann_single,
        # conf_ann_single_0008_relu, conf_ann_single_0008_selu,
        # conf_ann_single_0012_relu, conf_ann_single_0012_selu,
        # conf_ann_single_0016_relu, conf_ann_single_0016_selu,
        # conf_ann_single_0032_relu, conf_ann_single_0032_selu,
        # conf_ann_single_0064_relu, conf_ann_single_0064_selu,
        # conf_ann_single_0128_relu, conf_ann_single_0128_selu,
        # conf_ann_single_0256_relu, conf_ann_single_0256_selu,
        # conf_ann_single_0512_relu, conf_ann_single_0512_selu,
        # conf_ann_single_1024_relu, conf_ann_single_1024_selu,
        # conf_ann_single_2048_relu, conf_ann_single_2048_selu,
        # conf_lstm_standard,
        # conf_cnn_standard,
        # conf_resnet_standard,
        # conf_resnet_single,
        # conf_wavelet_500_standard_nn,
        # conf_wavelet_single_nn,
        # conf_new_wavelet_standard_nn,
        # conf_new_wavelet_single_nn,
        # conf_inception1d,
        # conf_inception1d_input256,
        # conf_inception1d_input512,
        # conf_inception1d_input1000,
        # conf_inception1d_no_residual,
        # conf_wavelet_standard_rf,
        # conf_wavelet_500_standard_nn,
        # conf_wavelet_single_nn,
        # conf_wavelet_single_rf,
        # conf_wavelet_single_xgb,
        # conf_wavelet_standard_xgb,
        # conf_ann_standard_0008_relu, conf_ann_standard_0008_selu,
        # conf_ann_standard_0012_relu, conf_ann_standard_0012_selu,
        # conf_ann_standard_0016_relu, conf_ann_standard_0016_selu,
        # conf_ann_standard_0032_relu, conf_ann_standard_0032_selu,
        # conf_ann_standard_0064_relu, conf_ann_standard_0064_selu,
        # conf_ann_standard_0128_relu, conf_ann_standard_0128_selu,
        # conf_ann_standard_0256_relu, conf_ann_standard_0256_selu,
        # conf_ann_standard_0512_relu, conf_ann_standard_0512_selu,
        # conf_ann_standard_1024_relu, conf_ann_standard_1024_selu,
        # conf_ann_standard_2048_relu, conf_ann_standard_2048_selu,
        # conf_lstm_lead1,
        # conf_lstm_standard,
        # conf_lstm_attn,
        conf_lenet12,
    ]
    experiments = [
        # ('exp0', 'all'),
        # ('exp1', 'diagnostic'),
        # ('exp1.1', 'subdiagnostic'),
        # ('exp1.1.1', 'superdiagnostic'),
        # ('exp2', 'form'),
        # ('exp3', 'rhythm'),
        ('exp4', 'priority')
    ]
    
    # for test (REMOVE BEFORE FULL RUN)
#     experiments = [experiments[-1]]
    
    for name, task in experiments:
        e = ECG_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        e.perform()
        e.evaluate(bootstrap_eval=True, dumped_bootstraps=False)
        
    # for name, task in experiments:
    #     f = DEMO_Experiment(name, task, datafolder, outputfolder, models)
    #     f.prepare()
    #     f.perform()
    #     f.evaluate()
    
    
    # generate summary table
    utils.generate_summary_table()

if __name__ == '__main__':
    main()