import colorcet
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import scipy.io as sio
import pickle

import method_functions
import visualisation

# stop recursive loops when multiprocessing, allows training multiple networks in parallel
if __name__ == '__main__':

    # generate training dataset? if false, load from disk
    train_dataset_flag = False
    # train networks? if false, load from disk
    train_flag = False
    # generate testing dataset? if false, load from disk
    test_dataset_flag = False
    # test networks? if false, load from disk
    test_flag = False
    # generate plots?
    plot_flag = True
    # save figures? if false, just plot them
    save_plots = True
    # process in-vivo data? if false, load outputs from disk
    clinical_data_flag = False

    # harmonise network initialisations?
    transfer_flag = True
    # number of networks to train (with different random initialisations); if transfer_flag = True, only
    # the best network is used for weight initialisation
    n_nets = 16

    # set global colormap
    matplotlib.rc('image', cmap=colorcet.cm.bmy_r)
    # set global font size
    matplotlib.pyplot.rcParams.update({'font.size': 20})

    # set python rng seed
    random.seed(a=71293)
    # set numpy rng seed
    np.random.seed(19824)
    # use deterministic pytorch algorithms
    torch.use_deterministic_algorithms(True)
    # get path of main script
    script_dir = os.path.dirname(__file__)

    # SNRs to investigate
    SNR_sweep = [30, 15]

    # generative model; check method_functions.generate_signal to implement other models
    model_name = 'ivim'
    # true noise type; check method_functions.add_noise to implement other noise types
    noise_type = 'rician'
    # acquisition protocol; check method_functions.get_sampling_scheme to implement other acquisition schemes
    sampling_distribution = 'ivim_10'
    sampling_scheme = method_functions.get_sampling_scheme(sampling_distribution)
    # network architecture; check method_functions.train_general_network to implement other network architectures
    network_arch = 'NN_narrow'
    criterion = nn.MSELoss()

    # total number of simulated datapoints, split between training and validation
    dataset_size = 100000
    # extent of parameter space and relative loss weighting of model parameters
    parameter_range, parameter_loss_scaling = method_functions.get_parameter_scaling(model_name, sampling_distribution)

    # generate folder structure
    method_functions.create_directories(script_dir=script_dir, network_arch=network_arch, model_name=model_name,
                                        noise_type=noise_type, sampling_distribution=sampling_distribution)

    # consider each SNR value in turn
    for SNR in SNR_sweep:
        if SNR == 30:
            SNR_alt = 15
        else:
            SNR_alt = 30

        # generate/load training + validation data
        training_data, validation_data = \
            method_functions.generate_training_data(train_dataset_flag=train_dataset_flag,
                                                    SNR=SNR,
                                                    script_dir=script_dir,
                                                    model_name=model_name,
                                                    noise_type=noise_type,
                                                    sampling_distribution=sampling_distribution,
                                                    dataset_size=dataset_size,
                                                    SNR_alt=SNR_alt,
                                                    parameter_range=parameter_range,
                                                    sampling_scheme=sampling_scheme)

        # train/load trained networks
        net_supervised_groundtruth, net_supervised_mle, net_supervised_mle_approx, net_selfsupervised, \
            net_supervised_hybrid, net_supervised_mle_weighted_f, \
            net_supervised_mle_weighted_Dslow, net_supervised_mle_weighted_Dfast = \
            method_functions.networks(train_flag=train_flag,
                                      SNR=SNR,
                                      transfer_flag=transfer_flag,
                                      script_dir=script_dir,
                                      network_arch=network_arch,
                                      model_name=model_name,
                                      noise_type=noise_type,
                                      sampling_distribution=sampling_distribution,
                                      dataset_size=dataset_size,
                                      SNR_alt=SNR_alt,
                                      n_nets=n_nets,
                                      training_data=training_data,
                                      validation_data=validation_data,
                                      parameter_loss_scaling=parameter_loss_scaling,
                                      criterion=criterion)

        # generate/load uniformly-distributed testing data
        test_data, test_mle_groundtruth, test_mle_mean = \
            method_functions.test_data_uniform(test_dataset_flag=test_dataset_flag,
                                               SNR=SNR,
                                               model_name=model_name,
                                               script_dir=script_dir,
                                               noise_type=noise_type,
                                               sampling_distribution=sampling_distribution,
                                               dataset_size=dataset_size,
                                               training_data=training_data)

        # evaluate network test performance on uniformly-distributed testing data
        test_mle_groundtruth, test_mle_mean, test_supervised_groundtruth, test_supervised_mle, \
            test_supervised_mle_approx, test_selfsupervised, \
            test_hybrid, test_mle_weighted_f, test_mle_weighted_Dslow, test_mle_weighted_Dfast = \
            method_functions.evaluate_methods(test_flag=test_flag,
                                              SNR=SNR,
                                              model_name=model_name,
                                              test_data=test_data,
                                              sampling_distribution=sampling_distribution,
                                              script_dir=script_dir,
                                              noise_type=noise_type,
                                              dataset_size=dataset_size,
                                              network_arch=network_arch,
                                              test_mle_groundtruth=test_mle_groundtruth,
                                              test_mle_mean=test_mle_mean,
                                              net_supervised_groundtruth=net_supervised_groundtruth,
                                              net_supervised_mle=net_supervised_mle,
                                              net_supervised_mle_approx=net_supervised_mle_approx,
                                              net_selfsupervised=net_selfsupervised,
                                              net_supervised_hybrid=net_supervised_hybrid,
                                              net_supervised_mle_weighted_f=net_supervised_mle_weighted_f,
                                              net_supervised_mle_weighted_Dslow=net_supervised_mle_weighted_Dslow,
                                              net_supervised_mle_weighted_Dfast=net_supervised_mle_weighted_Dfast)

        # generate/load clinically-distributed testing data
        test_clinical_real_mle_mean, test_clinical_real_supervised_groundtruth, test_clinical_real_supervised_mle, \
            test_clinical_real_supervised_mle_approx, test_clinical_real_selfsupervised, test_clinical_synth_mle_mean, \
            test_clinical_synth_supervised_groundtruth, test_clinical_synth_supervised_mle, \
            test_clinical_synth_supervised_mle_approx, test_clinical_synth_selfsupervised, clinical_groundtruth_masked = \
            method_functions.test_data_clinical(clinical_data_flag=clinical_data_flag,
                                                script_dir=script_dir,
                                                model_name=model_name,
                                                noise_type=noise_type,
                                                sampling_scheme=sampling_scheme,
                                                sampling_distribution=sampling_distribution,
                                                SNR=SNR,
                                                net_supervised_groundtruth=net_supervised_groundtruth,
                                                net_supervised_mle=net_supervised_mle,
                                                net_supervised_mle_approx=net_supervised_mle_approx,
                                                net_selfsupervised=net_selfsupervised)
        # generate plots
        if plot_flag:
            visualisation.generate_plots(save_plots=save_plots,
                                         network_arch=network_arch,
                                         dataset_size=dataset_size,
                                         training_data=training_data,
                                         test_data=test_data,
                                         SNR=SNR,
                                         script_dir=script_dir,
                                         parameter_range=parameter_range,
                                         clinical_groundtruth_masked=clinical_groundtruth_masked,
                                         test_mle_mean=test_mle_mean,
                                         test_supervised_groundtruth=test_supervised_groundtruth,
                                         test_supervised_mle=test_supervised_mle,
                                         test_supervised_mle_approx=test_supervised_mle_approx,
                                         test_selfsupervised=test_selfsupervised,
                                         test_hybrid=test_hybrid,
                                         test_supervised_mle_weighted_f=test_mle_weighted_f,
                                         test_supervised_mle_weighted_Dslow=test_mle_weighted_Dslow,
                                         test_supervised_mle_weighted_Dfast=test_mle_weighted_Dfast,
                                         test_clinical_real_mle_mean=test_clinical_real_mle_mean,
                                         test_clinical_real_supervised_mle=test_clinical_real_supervised_mle,
                                         test_clinical_real_selfsupervised=test_clinical_real_selfsupervised,
                                         test_clinical_synth_mle_mean=test_clinical_synth_mle_mean,
                                         test_clinical_synth_supervised_mle=test_clinical_synth_supervised_mle,
                                         test_clinical_synth_selfsupervised=test_clinical_synth_selfsupervised,
                                         test_clinical_synth_supervised_mle_approx=test_clinical_synth_supervised_mle_approx,
                                         test_clinical_synth_supervised_groundtruth=test_clinical_synth_supervised_groundtruth,
                                         test_clinical_real_supervised_mle_approx=test_clinical_real_supervised_mle_approx,
                                         test_clinical_real_supervised_groundtruth=test_clinical_real_supervised_groundtruth,
                                         fig_2=True,
                                         fig_3=True,
                                         fig_4=True,
                                         fig_5=True,
                                         fig_6=True,
                                         fig_7=True,
                                         fig_8=True,
                                         fig_9=True,
                                         fig_10=True,
                                         fig_11=True,
                                         fig_12=True)
    #
