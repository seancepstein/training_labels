import method_functions, visualisation
import colorcet, matplotlib, pickle, torch, random, os, pathlib
import numpy as np
import torch.nn as nn

# stop recursive loops when multiprocessing, allows training multiple networks in parallel
if __name__ == '__main__':

    # generate training dataset? if false, load from disk
    train_dataset_flag = False
    # train networks? if false, load from disk
    train_flag = True
    # generate testing dataset? if false, load from disk
    test_dataset_flag = True
    # test networks? if false, load from disk
    test_flag = True
    # generate plots?
    plot_flag = True
    # save figures? if not, just plot them
    save_plots = True

    # determine whether to train/test each network type
    supervised_groundtruth_flag = True
    supervised_mle_flag = True
    supervised_mle_approx_flag = True
    selfsupervised_flag = True
    weighted_flag = True
    hybrid_flag = True

    # determine whether to initialise network weights from supervised_groundtruth network
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
    sampling_distribution = 'ivim_12_extended'
    sampling_scheme = method_functions.get_sampling_scheme(sampling_distribution)
    # network architecture; check method_functions.train_general_network to implement other network architectures
    network_arch = 'NN_narrow'
    criterion = nn.MSELoss()

    # total number of simulated datapoints, split between training and validation
    dataset_size = 100000
    # extent of parameter space and relative loss weighting of model parameters
    parameter_range, parameter_loss_scaling = method_functions.getParameterScaling(model_name)

    # generate folder structure
    method_functions.create_directories(script_dir=script_dir, network_arch=network_arch, model_name=model_name,
                                        noise_type=noise_type, sampling_distribution=sampling_distribution)

    # consider each SNR value in turn
    for SNR in SNR_sweep:
        if SNR == SNR_sweep[0]:
            SNR_alt = SNR_sweep[1]
        else:
            SNR_alt = SNR_sweep[0]

        # generate training dataset
        if train_dataset_flag:
            # harmonise noisefree training data across SNR
            if SNR == SNR_sweep[1]:

                # load training and validation datasets from disk
                temp_file = open(os.path.join(script_dir,
                                              'data/train/{}/{}/{}/train_data_{}_{}.pkl'.format(model_name,
                                                                                                noise_type,
                                                                                                sampling_distribution,
                                                                                                dataset_size,
                                                                                                SNR_alt)), 'rb')
                training_data_alt = pickle.load(temp_file)
                temp_file.close()

                temp_file = open(os.path.join(script_dir,
                                              'data/train/{}/{}/{}/val_data_{}_{}.pkl'.format(model_name,
                                                                                              noise_type,
                                                                                              sampling_distribution,
                                                                                              dataset_size,
                                                                                              SNR_alt)), 'rb')
                validation_data_alt = pickle.load(temp_file)
                temp_file.close()

                imported_data_train = training_data_alt
                imported_data_val = validation_data_alt
            else:
                imported_data_train = False
                imported_data_val = False

            # generate training and validation datasets and save to disk
            training_data, validation_data = \
                method_functions.create_dataset(script_dir=script_dir, model_name=model_name,
                                                parameter_distribution='uniform', parameter_range=parameter_range,
                                                sampling_scheme=sampling_scheme,
                                                sampling_distribution=sampling_distribution, noise_type=noise_type,
                                                SNR=SNR, dataset_size=dataset_size,
                                                imported_data_train=imported_data_train,
                                                imported_data_val=imported_data_val)

        else:
            # load training and validation datasets from disk
            temp_file = open(os.path.join(script_dir,
                                          'data/train/{}/{}/{}/train_data_{}_{}.pkl'.format(model_name,
                                                                                            noise_type,
                                                                                            sampling_distribution,
                                                                                            dataset_size,
                                                                                            SNR)), 'rb')
            training_data = pickle.load(temp_file)
            temp_file.close()

            temp_file = open(os.path.join(script_dir,
                                          'data/train/{}/{}/{}/val_data_{}_{}.pkl'.format(model_name,
                                                                                          noise_type,
                                                                                          sampling_distribution,
                                                                                          dataset_size,
                                                                                          SNR)), 'rb')
            validation_data = pickle.load(temp_file)
            temp_file.close()

        # train networks
        if train_flag:

            # harmonise starting network state across SNRs
            if transfer_flag and SNR == SNR_sweep[1]:
                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/supervised_groundtruth_{}_{}.pkl'.format(network_arch,
                                                                                              model_name,
                                                                                              noise_type,
                                                                                              sampling_distribution,
                                                                                              dataset_size,
                                                                                              SNR_alt)), 'rb')
                net_supervised_groundtruth_alt = pickle.load(temp_file)
                temp_file.close()

                state_dictionary = net_supervised_groundtruth_alt.best_network.state_dict()

            if supervised_groundtruth_flag:

                if transfer_flag and SNR == SNR_sweep[1]:

                    net_supervised_groundtruth = method_functions.train_general_network(
                        model_name=training_data.model_name,
                        network_type='supervised_parameters',
                        architecture=network_arch,
                        n_nets=n_nets,
                        sampling_scheme=training_data.sampling_scheme,
                        training_loader=training_data.dataloader_supervised_groundtruth,
                        validation_loader=validation_data.dataloader_supervised_groundtruth,
                        method_name='supervised_groundtruth',
                        ignore_validation=False,
                        validation_condition=100,
                        training_split=training_data.train_split,
                        validation_split=training_data.val_split,
                        normalisation_factor=parameter_loss_scaling,
                        script_dir=script_dir,
                        network_arch=network_arch,
                        noise_type=noise_type,
                        sampling_distribution=sampling_distribution,
                        dataset_size=dataset_size,
                        SNR=SNR,
                        criterion=criterion,
                        transfer_learning=transfer_flag,
                        state_dictionary=state_dictionary)

                else:
                    net_supervised_groundtruth = method_functions.train_general_network(
                        model_name=training_data.model_name,
                        network_type='supervised_parameters',
                        architecture=network_arch,
                        n_nets=n_nets,
                        sampling_scheme=training_data.sampling_scheme,
                        training_loader=training_data.dataloader_supervised_groundtruth,
                        validation_loader=validation_data.dataloader_supervised_groundtruth,
                        method_name='supervised_groundtruth',
                        ignore_validation=False,
                        validation_condition=100,
                        training_split=training_data.train_split,
                        validation_split=training_data.val_split,
                        normalisation_factor=parameter_loss_scaling,
                        script_dir=script_dir,
                        network_arch=network_arch,
                        noise_type=noise_type,
                        sampling_distribution=sampling_distribution,
                        dataset_size=dataset_size,
                        SNR=SNR,
                        criterion=criterion,
                        transfer_learning=False,
                        state_dictionary=None)

                    state_dictionary = net_supervised_groundtruth.best_network.state_dict()

            if supervised_mle_flag:
                net_supervised_mle = method_functions.train_general_network(
                    model_name=training_data.model_name,
                    network_type='supervised_parameters',
                    architecture=network_arch,
                    n_nets=n_nets,
                    sampling_scheme=training_data.sampling_scheme,
                    training_loader=training_data.dataloader_supervised_mle,
                    validation_loader=validation_data.dataloader_supervised_mle,
                    method_name='supervised_mle',
                    ignore_validation=False,
                    validation_condition=100,
                    training_split=training_data.train_split,
                    validation_split=training_data.val_split,
                    normalisation_factor=parameter_loss_scaling,
                    script_dir=script_dir,
                    network_arch=network_arch,
                    noise_type=noise_type,
                    sampling_distribution=sampling_distribution,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    criterion=criterion,
                    transfer_learning=transfer_flag,
                    state_dictionary=state_dictionary)

            if supervised_mle_approx_flag:
                net_supervised_mle_approx = method_functions.train_general_network(
                    model_name=training_data.model_name,
                    network_type='supervised_parameters',
                    architecture=network_arch,
                    n_nets=n_nets,
                    sampling_scheme=training_data.sampling_scheme,
                    training_loader=training_data.dataloader_supervised_mle_approx,
                    validation_loader=validation_data.dataloader_supervised_mle_approx,
                    method_name='supervised_mle_approx',
                    ignore_validation=False,
                    validation_condition=100,
                    training_split=training_data.train_split,
                    validation_split=training_data.val_split,
                    normalisation_factor=parameter_loss_scaling,
                    script_dir=script_dir,
                    network_arch=network_arch,
                    noise_type=noise_type,
                    sampling_distribution=sampling_distribution,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    criterion=criterion,
                    transfer_learning=transfer_flag,
                    state_dictionary=state_dictionary)

            if selfsupervised_flag:
                net_selfsupervised = method_functions.train_general_network(
                    model_name=training_data.model_name,
                    network_type='selfsupervised',
                    architecture=network_arch,
                    n_nets=n_nets,
                    sampling_scheme=training_data.sampling_scheme,
                    training_loader=training_data.dataloader_supervised_groundtruth,
                    validation_loader=validation_data.dataloader_supervised_groundtruth,
                    method_name='selfsupervised',
                    ignore_validation=False,
                    validation_condition=100,
                    training_split=training_data.train_split,
                    validation_split=training_data.val_split,
                    normalisation_factor=parameter_loss_scaling,
                    script_dir=script_dir,
                    network_arch=network_arch,
                    noise_type=noise_type,
                    sampling_distribution=sampling_distribution,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    criterion=criterion,
                    transfer_learning=transfer_flag,
                    state_dictionary=state_dictionary)

            if weighted_flag:
                net_supervised_mle_weighted_f = method_functions.train_general_network(
                    model_name=training_data.model_name,
                    network_type='supervised_parameters',
                    architecture=network_arch,
                    n_nets=n_nets,
                    sampling_scheme=training_data.sampling_scheme,
                    training_loader=training_data.dataloader_supervised_mle,
                    validation_loader=validation_data.dataloader_supervised_mle,
                    method_name='supervised_mle_weighted_f',
                    ignore_validation=False,
                    validation_condition=100,
                    training_split=training_data.train_split,
                    validation_split=training_data.val_split,
                    normalisation_factor=[1e6, 1, 1e6, 1e6],
                    script_dir=script_dir,
                    network_arch=network_arch,
                    noise_type=noise_type,
                    sampling_distribution=sampling_distribution,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    criterion=criterion,
                    transfer_learning=transfer_flag,
                    state_dictionary=state_dictionary)
                net_supervised_mle_weighted_Dslow = method_functions.train_general_network(
                    model_name=training_data.model_name,
                    network_type='supervised_parameters',
                    architecture=network_arch,
                    n_nets=n_nets,
                    sampling_scheme=training_data.sampling_scheme,
                    training_loader=training_data.dataloader_supervised_mle,
                    validation_loader=validation_data.dataloader_supervised_mle,
                    method_name='supervised_mle_weighted_Dslow',
                    ignore_validation=False,
                    validation_condition=100,
                    training_split=training_data.train_split,
                    validation_split=training_data.val_split,
                    normalisation_factor=[1e6, 1e6, 1, 1e6],
                    script_dir=script_dir,
                    network_arch=network_arch,
                    noise_type=noise_type,
                    sampling_distribution=sampling_distribution,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    criterion=criterion,
                    transfer_learning=transfer_flag,
                    state_dictionary=state_dictionary)
                net_supervised_mle_weighted_Dfast = method_functions.train_general_network(
                    model_name=training_data.model_name,
                    network_type='supervised_parameters',
                    architecture=network_arch,
                    n_nets=n_nets,
                    sampling_scheme=training_data.sampling_scheme,
                    training_loader=training_data.dataloader_supervised_mle,
                    validation_loader=validation_data.dataloader_supervised_mle,
                    method_name='supervised_mle_weighted_Dfast',
                    ignore_validation=False,
                    validation_condition=100,
                    training_split=training_data.train_split,
                    validation_split=training_data.val_split,
                    normalisation_factor=[1e6, 1e6, 1e6, 1],
                    script_dir=script_dir,
                    network_arch=network_arch,
                    noise_type=noise_type,
                    sampling_distribution=sampling_distribution,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    criterion=criterion,
                    transfer_learning=transfer_flag,
                    state_dictionary=state_dictionary)

            if hybrid_flag:
                net_supervised_hybrid = method_functions.train_general_network(
                    model_name=training_data.model_name,
                    network_type='hybrid',
                    architecture=network_arch,
                    n_nets=n_nets,
                    sampling_scheme=training_data.sampling_scheme,
                    training_loader=training_data.dataloader_hybrid,
                    validation_loader=validation_data.dataloader_hybrid,
                    method_name='supervised_hybrid',
                    ignore_validation=False,
                    validation_condition=100,
                    training_split=training_data.train_split,
                    validation_split=training_data.val_split,
                    normalisation_factor=parameter_loss_scaling,
                    script_dir=script_dir,
                    network_arch=network_arch,
                    noise_type=noise_type,
                    sampling_distribution=sampling_distribution,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    criterion=criterion,
                    transfer_learning=transfer_flag,
                    state_dictionary=state_dictionary)

        # load networks from disk
        else:

            if supervised_groundtruth_flag:
                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/supervised_groundtruth_{}_{}.pkl'.format(network_arch, model_name,
                                                                                              noise_type,
                                                                                              sampling_distribution,
                                                                                              dataset_size, SNR)), 'rb')
                net_supervised_groundtruth = pickle.load(temp_file)
                temp_file.close()

            if supervised_mle_flag:
                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/supervised_mle_{}_{}.pkl'.format(network_arch, model_name,
                                                                                      noise_type,
                                                                                      sampling_distribution,
                                                                                      dataset_size, SNR)), 'rb')
                net_supervised_mle = pickle.load(temp_file)
                temp_file.close()

            if supervised_mle_approx_flag:
                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/supervised_mle_approx_{}_{}.pkl'.format(network_arch, model_name,
                                                                                             noise_type,
                                                                                             sampling_distribution,
                                                                                             dataset_size, SNR)), 'rb')
                net_supervised_mle_approx = pickle.load(temp_file)
                temp_file.close()

            if selfsupervised_flag:
                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/selfsupervised_{}_{}.pkl'.format(network_arch, model_name,
                                                                                      noise_type, sampling_distribution,
                                                                                      dataset_size, SNR)), 'rb')
                net_selfsupervised = pickle.load(temp_file)
                temp_file.close()

            if hybrid_flag:
                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/supervised_hybrid_{}_{}.pkl'.format(network_arch, model_name,
                                                                                         noise_type,
                                                                                         sampling_distribution,
                                                                                         dataset_size, SNR)), 'rb')
                net_supervised_hybrid = pickle.load(temp_file)
                temp_file.close()

            if weighted_flag:
                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/supervised_mle_weighted_f_{}_{}.pkl'.format(network_arch,
                                                                                                 model_name,
                                                                                                 noise_type,
                                                                                                 sampling_distribution,
                                                                                                 dataset_size, SNR)),
                    'rb')
                net_supervised_mle_weighted_f = pickle.load(temp_file)
                temp_file.close()

                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/supervised_mle_weighted_Dslow_{}_{}.pkl'.format(network_arch,
                                                                                                     model_name,
                                                                                                     noise_type,
                                                                                                     sampling_distribution,
                                                                                                     dataset_size,
                                                                                                     SNR)),
                    'rb')
                net_supervised_mle_weighted_Dslow = pickle.load(temp_file)
                temp_file.close()

                temp_file = open(
                    os.path.join(script_dir,
                                 'models/{}/{}/{}/{}/supervised_mle_weighted_Dfast_{}_{}.pkl'.format(network_arch,
                                                                                                     model_name,
                                                                                                     noise_type,
                                                                                                     sampling_distribution,
                                                                                                     dataset_size,
                                                                                                     SNR)),
                    'rb')
                net_supervised_mle_weighted_Dfast = pickle.load(temp_file)
                temp_file.close()

        if test_dataset_flag:

            # generate test dataset and compute MLE baseline
            test_mle_uniform, test_data_uniform, test_mle_uniform_approx = \
                method_functions.create_test_data(
                    training_data=training_data,
                    n_repeats=500,
                    n_sampling=[1, 10, 10, 10],
                    extent_scaling=0.0,
                    model_name=model_name,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    sampling_distribution=sampling_distribution,
                    dataset_size=dataset_size,
                    SNR=SNR)

        else:
            # load test dataset + MLE baseline from disk
            temp_file = open(os.path.join(script_dir,
                                          'data/test/{}/{}/{}/test_data_uniform_{}_{}.pkl'.format(model_name,
                                                                                                  noise_type,
                                                                                                  sampling_distribution,
                                                                                                  dataset_size,
                                                                                                  SNR)), 'rb')
            test_data_uniform = pickle.load(temp_file)
            temp_file.close()

            temp_file = open(os.path.join(script_dir,
                                          'data/test/{}/{}/{}/test_MLE_uniform_{}_{}.pkl'.format(model_name,
                                                                                                 noise_type,
                                                                                                 sampling_distribution,
                                                                                                 dataset_size,
                                                                                                 SNR)), 'rb')
            test_mle_uniform = pickle.load(temp_file)
            temp_file.close()

            temp_file = open(os.path.join(script_dir,
                                          'data/test/{}/{}/{}/test_MLE_uniform_approx_{}_{}.pkl'.format(model_name,
                                                                                                        noise_type,
                                                                                                        sampling_distribution,
                                                                                                        dataset_size,
                                                                                                        SNR)), 'rb')
            test_mle_uniform_approx = pickle.load(temp_file)
            temp_file.close()

        # evaluate network test performance
        if test_flag:

            # test mle labels
            test_mle_uniform = method_functions.test_performance_wrapper(
                network_type='mle',
                model_name=model_name,
                network=None,
                test_data=test_data_uniform,
                sampling_distribution=sampling_distribution,
                script_dir=script_dir,
                noise_type=noise_type,
                dataset_size=dataset_size,
                SNR=SNR,
                network_arch=network_arch,
                mle_flag=True,
                mle_data=test_mle_uniform)

            # compute test performance of networks
            if supervised_groundtruth_flag:
                test_supervised_groundtruth_uniform = method_functions.test_performance_wrapper(
                    network_type='supervised_groundtruth',
                    model_name=model_name,
                    network=net_supervised_groundtruth,
                    test_data=test_data_uniform,
                    sampling_distribution=sampling_distribution,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    network_arch=network_arch)

            if supervised_mle_flag:
                test_supervised_mle_uniform = method_functions.test_performance_wrapper(
                    network_type='supervised_mle',
                    model_name=model_name,
                    network=net_supervised_mle,
                    test_data=test_data_uniform,
                    sampling_distribution=sampling_distribution,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    network_arch=network_arch)

            if supervised_mle_approx_flag:
                test_supervised_mle_approx_uniform = method_functions.test_performance_wrapper(
                    network_type='supervised_mle_approx',
                    model_name=model_name,
                    network=net_supervised_mle_approx,
                    test_data=test_data_uniform,
                    sampling_distribution=sampling_distribution,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    network_arch=network_arch)

            if selfsupervised_flag:
                test_unsupervised_uniform = method_functions.test_performance_wrapper(
                    network_type='unsupervised',
                    model_name=model_name,
                    network=net_selfsupervised,
                    test_data=test_data_uniform,
                    sampling_distribution=sampling_distribution,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    network_arch=network_arch)

            if hybrid_flag:
                test_hybrid_uniform = method_functions.test_performance_wrapper(
                    network_type='hybrid',
                    model_name=model_name,
                    network=net_supervised_hybrid,
                    test_data=test_data_uniform,
                    sampling_distribution=sampling_distribution,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    network_arch=network_arch)
            if weighted_flag:
                test_mle_weighted_f_uniform = method_functions.test_performance_wrapper(
                    network_type='mle_weighted_f',
                    model_name=model_name,
                    network=net_supervised_mle_weighted_f,
                    test_data=test_data_uniform,
                    sampling_distribution=sampling_distribution,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    network_arch=network_arch)
                test_mle_weighted_Dslow_uniform = method_functions.test_performance_wrapper(
                    network_type='mle_weighted_Dslow',
                    model_name=model_name,
                    network=net_supervised_mle_weighted_Dslow,
                    test_data=test_data_uniform,
                    sampling_distribution=sampling_distribution,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    network_arch=network_arch)
                test_mle_weighted_Dfast_uniform = method_functions.test_performance_wrapper(
                    network_type='mle_weighted_Dfast',
                    model_name=model_name,
                    network=net_supervised_mle_weighted_Dfast,
                    test_data=test_data_uniform,
                    sampling_distribution=sampling_distribution,
                    script_dir=script_dir,
                    noise_type=noise_type,
                    dataset_size=dataset_size,
                    SNR=SNR,
                    network_arch=network_arch)

        else:
            # load test performance from disk

            temp_file = open(os.path.join(script_dir,
                                          'results/{}/{}/{}/{}/test_mle_{}_{}.pkl'.format(
                                              network_arch, model_name, noise_type, sampling_distribution,
                                              dataset_size, SNR)), 'rb')
            test_mle_uniform = pickle.load(temp_file)
            temp_file.close()

            if supervised_groundtruth_flag:
                temp_file = open(os.path.join(script_dir,
                                              'results/{}/{}/{}/{}/test_supervised_groundtruth_{}_{}.pkl'.format(
                                                  network_arch, model_name, noise_type, sampling_distribution,
                                                  dataset_size, SNR)), 'rb')
                test_supervised_groundtruth_uniform = pickle.load(temp_file)
                temp_file.close()

            if supervised_mle_flag:
                temp_file = open(os.path.join(script_dir,
                                              'results/{}/{}/{}/{}/test_supervised_mle_{}_{}.pkl'.format(
                                                  network_arch, model_name, noise_type, sampling_distribution,
                                                  dataset_size, SNR)), 'rb')
                test_supervised_mle_uniform = pickle.load(temp_file)
                temp_file.close()

            if supervised_mle_approx_flag:
                temp_file = open(os.path.join(script_dir,
                                              'results/{}/{}/{}/{}/test_supervised_mle_approx_{}_{}.pkl'.format(
                                                  network_arch, model_name, noise_type, sampling_distribution,
                                                  dataset_size, SNR)), 'rb')
                test_supervised_mle_approx_uniform = pickle.load(temp_file)
                temp_file.close()

            if selfsupervised_flag:
                temp_file = open(os.path.join(script_dir,
                                              'results/{}/{}/{}/{}/test_unsupervised_{}_{}.pkl'.format(
                                                  network_arch, model_name, noise_type, sampling_distribution,
                                                  dataset_size, SNR)), 'rb')
                test_unsupervised_uniform = pickle.load(temp_file)
                temp_file.close()

            if hybrid_flag:
                temp_file = open(os.path.join(script_dir,
                                              'results/{}/{}/{}/{}/test_hybrid_{}_{}.pkl'.format(
                                                  network_arch, model_name, noise_type, sampling_distribution,
                                                  dataset_size, SNR)), 'rb')
                test_hybrid_uniform = pickle.load(temp_file)
                temp_file.close()

            if weighted_flag:
                temp_file = open(os.path.join(script_dir,
                                              'results/{}/{}/{}/{}/test_mle_weighted_f_{}_{}.pkl'.format(
                                                  network_arch, model_name, noise_type, sampling_distribution,
                                                  dataset_size, SNR)), 'rb')
                test_mle_weighted_f_uniform = pickle.load(temp_file)
                temp_file.close()

                temp_file = open(os.path.join(script_dir,
                                              'results/{}/{}/{}/{}/test_mle_weighted_Dslow_{}_{}.pkl'.format(
                                                  network_arch, model_name, noise_type, sampling_distribution,
                                                  dataset_size, SNR)), 'rb')
                test_mle_weighted_Dslow_uniform = pickle.load(temp_file)
                temp_file.close()

                temp_file = open(os.path.join(script_dir,
                                              'results/{}/{}/{}/{}/test_mle_weighted_Dfast_{}_{}.pkl'.format(
                                                  network_arch, model_name, noise_type, sampling_distribution,
                                                  dataset_size, SNR)), 'rb')
                test_mle_weighted_Dfast_uniform = pickle.load(temp_file)
                temp_file.close()

        if plot_flag:
            # plot figures
            visualisation.generate_plots(save_plots=save_plots,
                                         model_name=model_name,
                                         network_arch=network_arch,
                                         dataset_size=dataset_size,
                                         training_data=training_data,
                                         test_data_uniform=test_data_uniform,
                                         test_mle_uniform=test_mle_uniform,
                                         test_supervised_groundtruth=test_supervised_groundtruth_uniform,
                                         test_supervised_mle=test_supervised_mle_uniform,
                                         test_supervised_mle_approx=test_supervised_mle_approx_uniform,
                                         test_selfsupervised=test_unsupervised_uniform,
                                         test_hybrid=test_hybrid_uniform,
                                         test_supervised_mle_weighted_f=test_mle_weighted_f_uniform,
                                         test_supervised_mle_weighted_Dslow=test_mle_weighted_Dslow_uniform,
                                         test_supervised_mle_weighted_Dfast=test_mle_weighted_Dfast_uniform,
                                         SNR=SNR,
                                         script_dir=script_dir,
                                         fig_2=True,
                                         fig_3=True,
                                         fig_4=True,
                                         fig_5=True,
                                         fig_6=True,
                                         fig_7=True,
                                         fig_8=True,
                                         fig_9=True,
                                         fig_10=True)
