import nibabel
import torch, pickle, random, itertools, os, pathlib, sys, time, datetime
import method_classes
import numpy as np
import numpy.matlib as matlib
import torch.optim as optim
import torch.utils.data as utils
import numpy.ma as ma
from multiprocessing import Pool
from scipy.optimize import minimize, Bounds
from scipy import special
from numpy.random import default_rng


def create_directories(script_dir, model_name, network_arch, noise_type, sampling_distribution):
    """ Function to create local directories to save data, networks, figures, and results to

        Inputs
        ------

        script_dir : string
            base directory of main script

        model_name : string
            name of signal model being fit

        network_arch : string
            network architecture, as defined in method_functions.train_general_network

        noise_type: string
            defined in method_functions.add_noise, name of noise type

         sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    pathlib.Path(os.path.join(script_dir, 'models/{}/{}/{}/{}'.format(network_arch,
                                                                      model_name,
                                                                      noise_type,
                                                                      sampling_distribution))).mkdir(parents=True,
                                                                                                     exist_ok=True)
    pathlib.Path(os.path.join(script_dir, 'results/{}/{}/{}/{}'.format(network_arch, model_name, noise_type,
                                                                       sampling_distribution))).mkdir(
        parents=True, exist_ok=True)
    pathlib.Path(os.path.join(script_dir, 'data/test/{}/{}/{}'.format(model_name,
                                                                      noise_type,
                                                                      sampling_distribution))).mkdir(
        parents=True, exist_ok=True)
    pathlib.Path(os.path.join(script_dir,
                              'figures/{}/{}/{}/{}'.format(network_arch, model_name, noise_type,
                                                           sampling_distribution))).mkdir(
        parents=True,
        exist_ok=True)


def get_sampling_scheme(sampling_distribution):
    """ Function to generate generative sampling scheme associated with a named sampling distribution

        Inputs
        ------

        sampling_distribution : string
            name of sampling distribution

        Outputs
        -------
        sampling_scheme : ndarray
            provides signal sampling scheme (independent variable values)

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    if sampling_distribution == 'ivim_12':
        sampling_scheme = np.array([0, 10, 20, 30, 40, 50, 80, 100, 150, 200, 400, 800]) * 1e-3
    elif sampling_distribution == 'ivim_12_extended':
        sampling_scheme = np.array([0, 10, 20, 30, 40, 50, 80, 100, 150, 200, 400, 800]) * 1e-3
    elif sampling_distribution == 'ivim_12_subset':
        sampling_scheme = np.array([0, 10, 20, 30, 40, 50, 80, 100, 150, 200, 400, 800]) * 1e-3
    elif sampling_distribution == 'ivim_5':
        sampling_scheme = np.array([0, 50, 100, 300, 600]) * 1e-3
    elif sampling_distribution == 'ivim_9':
        sampling_scheme = np.array([0, 10, 20, 40, 80, 100, 200, 400, 600]) * 1e-3
    elif sampling_distribution == 'ivim_160':
        sampling_scheme = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                                    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                                    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                    80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
                                    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                                    400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
                                    800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800,
                                    800]) * 1e-3
    elif sampling_distribution == 'ivim_10':
        sampling_scheme = np.array([0, 10, 20, 30, 50, 80, 100, 200, 400, 800]) * 1e-3
    else:
        sys.exit("Implement other sampling schemes in method_functions.get_sampling_scheme")
    return sampling_scheme


def get_parameter_scaling(model_name, sampling_distribution):
    """ Function to generate generative parameter range + distribution associated with a given signal model

        Inputs
        ------

        model_name : string
            name of signal model being fit

        sampling_distribution : string
            name of sampling distribution


        Outputs
        -------
        parameter_range : ndarray
            provides boundaries for parameter_distribution

        parameter_loss_scaling : ndarray
            relative weighting of each signal model parameter during supervised training; larger scaling
            results in lower weighting

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    if model_name == 'ivim' and sampling_distribution == 'ivim_12_extended':
        parameter_range = np.array([[0.8, 1.2],  # S0
                                    [0, 0.8],  # f
                                    [0, 4.0],  # Dslow
                                    [0, 80]])  # Dfast
        parameter_loss_scaling = [1, 0.25, 1.25, 30]
    elif model_name == 'ivim' and sampling_distribution == 'ivim_12':
        parameter_range = np.array([[0.8, 1.2],  # S0
                                    [0.1, 0.4],  # f
                                    [0.5, 2.5],  # Dslow
                                    [10, 150]])  # Dfast
        parameter_loss_scaling = [1, 0.25, 1.5, 80]
    elif model_name == 'ivim' and sampling_distribution == 'ivim_12_subset':
        parameter_range = np.array([[0.8, 1.2],  # S0
                                    [0.2, 0.3],  # f
                                    [1.0, 1.5],  # Dslow
                                    [20, 40]])  # Dfast
        parameter_loss_scaling = [1, 0.25, 1.25, 30]
    elif model_name == 'ivim' and sampling_distribution == 'ivim_5':
        parameter_range = np.array([[0.8, 1.2],  # S0
                                    [0.1, 0.4],  # f
                                    [0.5, 2.0],  # Dslow
                                    [10, 200]])  # Dfast
        parameter_loss_scaling = [1, 0.25, 1.25, 30]
    elif model_name == 'ivim' and sampling_distribution == 'ivim_9':
        parameter_range = np.array([[0.8, 1.2],  # S0
                                    [0.05, 0.5],  # f
                                    [0.01, 3.0],  # Dslow
                                    [10, 150]])  # Dfast
        parameter_loss_scaling = [1, 0.2, 1.505, 80]
    elif model_name == 'ivim' and sampling_distribution == 'ivim_160':
        parameter_range = np.array([[0.8, 1.2],  # S0
                                    [0.1, 0.5],  # f
                                    [0.4, 3.0],  # Dslow
                                    [10, 150]])  # Dfast
        parameter_loss_scaling = [1, 0.3, 1.70, 80]
    elif model_name == 'ivim' and sampling_distribution == 'ivim_10':
        parameter_range = np.array([[0.8, 1.2],  # S0
                                    [0.1, 0.5],  # f
                                    [0.4, 3.0],  # Dslow
                                    [10, 150]])  # Dfast
        parameter_loss_scaling = [1, 0.3, 1.70, 80]
    else:
        sys.exit("Implement other signal models here")

    return parameter_range, parameter_loss_scaling


def create_dataset(script_dir, model_name, parameter_distribution, parameter_range, sampling_scheme,
                   sampling_distribution, noise_type, SNR, dataset_size, training_split=0.8, validation_split=0.2,
                   batch_size=100, num_workers=0, imported_data_train=False, imported_data_val=False):
    """ Function to create training dataset

        Inputs
        ------
        script_dir : string
            base directory of main script

        model_name : string
            name of signal model being fit

        parameter_distribution : string
            name of distribution used to draw generative parameters for training, defined in
            method_functions.generate_label

        parameter_range : ndarray
            defined in method_functions.get_parameter_scaling, provides boundaries for parameter_distribution

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme (independent
            variable values)

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        dataset_size : int
            total number of signals generated for training and validation

        training_split : optional, float
            proportion of dataset_size allocated to training

        validation_split : optional, float
            proportion of dataset_size allocated to validation

        batch_size : optional, int
            defined in torch.utils.data, size of miniloader batch

        num_workers : optional, int
            defined in torch.utils.data, number of subprocesses used for data loading

        imported_data_train : optional, method_classes.Dataset
            pre-existing training dataset, used to harmonise noise-free signals across different SNR

        imported_data_val : optional, method_classes.Dataset
            pre-existing validation dataset, used to harmonise noise-free signals across different SNR

        Outputs
        -------
        training_dataset : method_classes.Dataset
            object containing training data and dataloaders

        validation_dataset : method_classes.Dataset
            object containing validation data and dataloaders

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    print('Generating noise-free data...')
    # if pre-existing noisefree data is provided, use that
    if imported_data_train:
        train_signal = imported_data_train.data_clean
        train_label_groundtruth = imported_data_train.label_groundtruth
    else:
        # generate noise-free training data
        train_signal, train_label_groundtruth = generate_signal(model_name=model_name,
                                                                parameter_distribution=parameter_distribution,
                                                                parameter_range=parameter_range,
                                                                sampling_scheme=sampling_scheme,
                                                                n_signals=int(dataset_size * training_split))
    if imported_data_val:
        val_signal = imported_data_val.data_clean
        val_label_groundtruth = imported_data_val.label_groundtruth
    else:
        # generate noise-free validation data
        val_signal, val_label_groundtruth = generate_signal(model_name=model_name,
                                                            parameter_distribution=parameter_distribution,
                                                            parameter_range=parameter_range,
                                                            sampling_scheme=sampling_scheme,
                                                            n_signals=int(dataset_size * validation_split))
    print('Generating noisy data...')
    # add noise to training data
    train_signal_noisy = add_noise(noise_type=noise_type,
                                   SNR=SNR,
                                   signal=train_signal)

    # add noise to validation data
    val_signal_noisy = add_noise(noise_type=noise_type,
                                 SNR=SNR,
                                 signal=val_signal)

    print('Computing MLE for noisy data using correct noise model...')
    # calculate MLE estimates of training data
    train_label_bestfit, train_loss_bestfit, train_label_bestfit_all = \
        traditional_fitting_create_data(model_name=model_name,
                                        signal=train_signal_noisy,
                                        sampling_scheme=sampling_scheme,
                                        labels_groundtruth=train_label_groundtruth,
                                        SNR=SNR,
                                        noise_type=noise_type,
                                        seed_mean=False)

    # calculate MLE estimates of validation data
    val_label_bestfit, val_loss_bestfit, val_label_bestfit_all = \
        traditional_fitting_create_data(model_name=model_name,
                                        signal=val_signal_noisy,
                                        sampling_scheme=sampling_scheme,
                                        labels_groundtruth=val_label_groundtruth,
                                        SNR=SNR,
                                        noise_type=noise_type,
                                        seed_mean=False)

    if model_name == 'ivim' and noise_type == 'rician':
        print('Computing MLE for noisy data using incorrect noise model...')
        # calculate MLE using gaussian noise model, despite rician noise
        train_label_bestfit_approx, train_loss_bestfit_approx, train_label_bestfit_all_approx = \
            traditional_fitting_create_data(model_name=model_name,
                                            signal=train_signal_noisy,
                                            sampling_scheme=sampling_scheme,
                                            labels_groundtruth=train_label_groundtruth,
                                            SNR=SNR,
                                            noise_type='gaussian',
                                            seed_mean=False)

        # calculate MLE using gaussian noise model, despite rician noise
        val_label_bestfit_approx, val_loss_bestfit_approx, val_label_bestfit_all_approx = \
            traditional_fitting_create_data(model_name=model_name,
                                            signal=val_signal_noisy,
                                            sampling_scheme=sampling_scheme,
                                            labels_groundtruth=val_label_groundtruth,
                                            SNR=SNR,
                                            noise_type='gaussian',
                                            seed_mean=False)

    else:
        train_label_bestfit_approx = None
        train_label_bestfit_all_approx = None
        train_loss_bestfit_approx = None
        val_label_bestfit_approx = None
        val_label_bestfit_all_approx = None
        val_loss_bestfit_approx = None

    # # identify non-spurious MLE labels
    # train_idx_filtered = ((train_label_bestfit[:, 1] > 0.01) & (train_label_bestfit[:, 1] < 0.5) & (
    #         train_label_bestfit[:, 2] > 0.01) & (train_label_bestfit[:, 2] < 4.0) & (
    #                               train_label_bestfit[:, 3] > 2.0) & (train_label_bestfit[:, 2] < 300))
    # val_idx_filtered = ((val_label_bestfit[:, 1] > 0.01) & (val_label_bestfit[:, 1] < 0.5) & (
    #         val_label_bestfit[:, 2] > 0.01) & (val_label_bestfit[:, 2] < 4.0) & (
    #                             val_label_bestfit[:, 3] > 2.0) & (val_label_bestfit[:, 2] < 300))
    # train_approx_idx_filtered = (
    #         (train_label_bestfit_approx[:, 1] > 0.01) & (train_label_bestfit_approx[:, 1] < 0.5) & (
    #         train_label_bestfit_approx[:, 2] > 0.01) & (train_label_bestfit_approx[:, 2] < 4.0) & (
    #                 train_label_bestfit_approx[:, 3] > 2.0) & (train_label_bestfit_approx[:, 2] < 300))
    # val_approx_idx_filtered = (
    #         (val_label_bestfit_approx[:, 1] > 0.01) & (val_label_bestfit_approx[:, 1] < 0.5) & (
    #         val_label_bestfit_approx[:, 2] > 0.01) & (val_label_bestfit_approx[:, 2] < 4.0) & (
    #                 val_label_bestfit_approx[:, 3] > 2.0) & (val_label_bestfit_approx[:, 2] < 300))

    # construct dataloader objects for each data/label combination
    train_dataloader_supervised_groundtruth = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(train_signal_noisy),
            torch.Tensor(train_label_groundtruth)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataloader_supervised_groundtruth = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(val_signal_noisy),
            torch.Tensor(val_label_groundtruth)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataloader_supervised_mle = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(train_signal_noisy),
            torch.Tensor(train_label_bestfit)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataloader_supervised_mle = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(val_signal_noisy),
            torch.Tensor(val_label_bestfit)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataloader_supervised_mle_approx = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(train_signal_noisy),
            torch.Tensor(train_label_bestfit_approx)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataloader_supervised_mle_approx = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(val_signal_noisy),
            torch.Tensor(val_label_bestfit_approx)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataloader_hybrid = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(train_signal_noisy),
            torch.Tensor(np.stack([train_label_bestfit,
                                   train_label_groundtruth], axis=2))),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataloader_hybrid = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(val_signal_noisy),
            torch.Tensor(np.stack(
                [val_label_bestfit, val_label_groundtruth],
                axis=2))),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    pathlib.Path(os.path.join(script_dir, 'data/train/{}/{}/{}'.format(model_name, noise_type,
                                                                       sampling_distribution,
                                                                       dataset_size,
                                                                       SNR))).mkdir(parents=True, exist_ok=True)

    # store training data in Dataset object
    training_dataset = method_classes.Dataset(model_name=model_name,
                                              parameter_distribution=parameter_distribution,
                                              parameter_range=parameter_range,
                                              sampling_scheme=sampling_scheme,
                                              noise_type=noise_type,
                                              SNR=SNR,
                                              n_signals=int(training_split * dataset_size),
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              data_clean=train_signal,
                                              data_noisy=train_signal_noisy,
                                              label_groundtruth=train_label_groundtruth,
                                              label_bestfit=train_label_bestfit,
                                              label_bestfit_all=train_label_bestfit_all,
                                              label_bestfit_approx=train_loss_bestfit_approx,
                                              label_bestfit_all_approx=train_label_bestfit_all_approx,
                                              loss_bestfit=train_loss_bestfit,
                                              loss_bestfit_approx=train_loss_bestfit_approx,
                                              dataloader_supervised_groundtruth=train_dataloader_supervised_groundtruth,
                                              dataloader_supervised_mle=train_dataloader_supervised_mle,
                                              dataloader_supervised_mle_approx=train_dataloader_supervised_mle_approx,
                                              dataloader_hybrid=train_dataloader_hybrid,
                                              train_split=training_split,
                                              val_split=validation_split,
                                              sampling_distribution=sampling_distribution)
    # save Dataset to disk
    temp_file = open(os.path.join(script_dir, 'data/train/{}/{}/{}/train_data_{}_{}.pkl'.format(model_name, noise_type,
                                                                                                sampling_distribution,
                                                                                                dataset_size,
                                                                                                SNR)), 'wb')
    pickle.dump(training_dataset, temp_file)
    temp_file.close()

    # store validation data in Dataset object
    validation_dataset = method_classes.Dataset(model_name=model_name,
                                                parameter_distribution=parameter_distribution,
                                                parameter_range=parameter_range,
                                                sampling_scheme=sampling_scheme,
                                                noise_type=noise_type,
                                                SNR=SNR,
                                                n_signals=int(validation_split * dataset_size),
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                data_clean=val_signal,
                                                data_noisy=val_signal_noisy,
                                                label_groundtruth=val_label_groundtruth,
                                                label_bestfit=val_label_bestfit,
                                                label_bestfit_all=val_label_bestfit_all,
                                                label_bestfit_approx=val_loss_bestfit_approx,
                                                label_bestfit_all_approx=val_label_bestfit_all_approx,
                                                loss_bestfit=val_loss_bestfit,
                                                loss_bestfit_approx=val_loss_bestfit_approx,
                                                dataloader_supervised_groundtruth=val_dataloader_supervised_groundtruth,
                                                dataloader_supervised_mle=val_dataloader_supervised_mle,
                                                dataloader_supervised_mle_approx=val_dataloader_supervised_mle_approx,
                                                dataloader_hybrid=val_dataloader_hybrid,
                                                train_split=training_split,
                                                val_split=validation_split,
                                                sampling_distribution=sampling_distribution)

    # save Dataset object to disk
    temp_file = open(os.path.join(script_dir, 'data/train/{}/{}/{}/val_data_{}_{}.pkl'.format(model_name, noise_type,
                                                                                              sampling_distribution,
                                                                                              dataset_size, SNR
                                                                                              )),
                     'wb')
    pickle.dump(validation_dataset, temp_file)
    temp_file.close()
    return training_dataset, validation_dataset


def generate_signal(model_name, parameter_distribution, parameter_range, sampling_scheme, n_signals,
                    generative_label=None):
    """ Function to generate noisefree signals

        Inputs
        ------
        model_name : string
            name of signal model being fit

        parameter_distribution : string
            name of distribution used to draw generative parameters for training, defined in
            method_functions.generate_label

        parameter_range : ndarray
            defined in method_functions.get_parameter_scaling, provides boundaries for
            parameter_distribution

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme (independent
            variable values)

        n_signals : int
            number of signals to generate

        generative_label : optional, ndarray
            supplied labels (i.e. model parameters) if not generating from scratch

        Outputs
        -------
        signal : ndarray
            noisefree signal

        label : ndarray
            groundtruth generative parameters

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # define mapping from model name to model generative function
    generative_dictionary = {
        'ivim': generate_IVIM
    }
    # obtain number of free parameters associated with model
    n_free_parameters = get_n_free_parameters(model_name)
    # allocate memory for signal object
    signal = np.zeros((n_signals, len(sampling_scheme)))
    # default behaviour, no generative labels supplied, labels generate at runtime
    if generative_label is None:
        # allocate memory for labels
        label = np.zeros((n_signals, n_free_parameters))
        # iterate over signals
        for i in range(n_signals):
            # generate and save label
            label[i, :] = generate_label(model_name, parameter_distribution, parameter_range)
            # synthesise signal from generated label
            signal[i, :] = generative_dictionary[model_name](label[i, :], sampling_scheme)
    # otherwise, use supplied labels
    else:
        label = generative_label
        # iterate over signals
        for i in range(n_signals):
            # synthesise signal from supplied label
            signal[i, :] = generative_dictionary[model_name](label[i, :], sampling_scheme)
    return signal, label


def generate_IVIM(label, sampling_scheme):
    """ Function to generate IVIM signal

        Inputs
        ------
        label : ndarray
            groundtruth generative parameters

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme
            (independent variable values)

        Outputs
        -------
        signal : ndarray
            noisefree signal

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # read model parameters from numpy label object
    s0 = label[0]
    f = label[1]
    dslow = label[2]
    dfast = label[3]
    # generate numpy signal from model parameters
    signal = s0 * (
            f * np.exp(np.dot(-sampling_scheme, dslow + dfast)) + (1 - f) * np.exp(np.dot(-sampling_scheme, dslow)))
    return signal


def generate_label(model_name, parameter_distribution, parameter_range):
    """ Function to compute generative model parameter

        Inputs
        ------
        model_name : string
            name of signal model being fit

        parameter_distribution : string
            name of distribution used to draw generative parameters for training, defined in
            method_functions.generate_label

        parameter_range : ndarray
            defined in method_functions.get_parameter_scaling, provides boundaries for
            parameter_distribution

        Outputs
        -------
        label : ndarray
            groundtruth generative parameters

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    # define mapping from parameter distribution name to sampling scheme
    distribution_dictionary = {
        'uniform': np.random.uniform
    }
    # obtain number of free parameters associated with model
    n_params = get_n_free_parameters(model_name)
    # allocate memory for label object
    label = np.zeros(n_params)
    # iterate over each parameters
    for parameter in range(n_params):
        # draw from parameter_range using parameter_distribution
        label[parameter] = distribution_dictionary[parameter_distribution](
            parameter_range[parameter, 0],
            parameter_range[parameter, 1])
    return label


def get_n_free_parameters(model_name):
    """ Function to lookup number of free parameters associated with a signal model

        Inputs
        ------
        model_name : string
            name of signal model being fit

        Outputs
        -------
        n_free_parameters : int
            number of signal model free parameters

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # define mapping from model name to number of free parameters
    free_params_dictionary = {
        'ivim': 4
    }
    # read from dictionary to obtain number of free model parameters
    n_free_parameters = free_params_dictionary[model_name]
    return n_free_parameters


def add_noise(noise_type, SNR, signal, clinical_flag=False):
    """ Function to add noise to signals

        Inputs
        ------

        noise_type: string
            name of noise type

        SNR : int
            signal-to-noise ratio

        signal : ndarray
            noisefree signal

        clinical_flag=False : optional, bool
            set to True if signal is clinical data

        Outputs
        -------
        signal_noisy : ndarray
            noisy signal

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # number of independent signals
    n_signal = signal.shape[0]
    if clinical_flag:
        #  number of repeats
        n_repeats = signal.shape[1]
        # number of sampling points per signal
        n_sampling = signal.shape[2]
        # standard deviation
        sigma = np.divide(1, SNR)
    else:
        # number of sampling points per signal
        n_sampling = signal.shape[1]
        # standard deviation
        sigma = 1 / SNR

    # add rician noise
    if noise_type == 'rician':
        if clinical_flag:
            # allocate memory to two signal channels
            signal_real = np.zeros([n_signal, n_repeats, n_sampling])
            signal_imag = np.zeros([n_signal, n_repeats, n_sampling])
            # run through all signals
            for signal_idx in range(n_signal):
                for repeat in range(n_repeats):
                    # generate real channel from signal + noise
                    signal_real[signal_idx, repeat, :] = signal[signal_idx, repeat, :] + np.random.normal(
                        scale=sigma[signal_idx], size=n_sampling)
                    # generate imaginary channel from only noise
                    signal_imag[signal_idx, repeat, :] = np.random.normal(scale=sigma[signal_idx], size=n_sampling)
        else:
            # allocate memory to two signal channels
            signal_real = np.zeros([n_signal, n_sampling])
            signal_imag = np.zeros([n_signal, n_sampling])
            # run through all signals
            for signal_idx in range(n_signal):
                # generate real channel from signal + noise
                signal_real[signal_idx, :] = signal[signal_idx, :] + np.random.normal(scale=sigma, size=n_sampling)
                # generate imaginary channel from only noise
                signal_imag[signal_idx, :] = np.random.normal(scale=sigma, size=n_sampling)

        # combine real and imaginary channels
        signal_noisy = np.sqrt(signal_real ** 2 + signal_imag ** 2)
    # add gaussian noise
    if noise_type == 'gaussian':
        if clinical_flag:
            signal_noisy = signal + np.random.normal(scale=sigma, size=(n_signal, n_repeats, n_sampling))
        else:
            signal_noisy = signal + np.random.normal(scale=sigma, size=(n_signal, n_sampling))
    return signal_noisy


def traditional_fitting(model_name, signal, sampling_scheme, labels_groundtruth, SNR, noise_type,
                        seed_mean=False, calculate_sigma=False, sigma_array=None, alternate_seed=None):
    """ Function to compute conventional MLE

        Inputs
        ------

        model_name : string
            name of signal model being fit

        signal : ndarray
            signal to fit model to

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme (independent
            variable values)

        labels_groundtruth : ndarray
            groundtruth generative parameters

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        seed_mean : optional, bool
            if True, seeds fitting with mean parameter values; if False, seeds with groundtruth values

        calculate_sigma : optional, bool
            if True, estimate signal standard deviation from b=0 independently for each model fit

        sigma_array : optional, ndarray
            standard deviation corresponding to each signal

        alternate_seed: optional, ndarray
            user-supplied seed used to initialise fitting

        Outputs
        -------
        labels_bestfit : ndarray
            best fit MLE parameters

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    # generate empty object to store bestfit parameters
    labels_bestfit = np.zeros_like(labels_groundtruth)
    loss_bestfit = np.zeros((labels_groundtruth.shape[0], 1))
    # number of signals being fit
    # n_signals = 2000
    n_signals = signal.shape[0]
    if model_name == 'ivim':
        if seed_mean:
            # determine fitting seed, set to mean parameter value
            seed = np.ones_like(labels_groundtruth) * labels_groundtruth.mean(axis=0)
        else:
            seed = labels_groundtruth

        # set upper and lower bounds for fitting
        # bounds = Bounds(lb=np.array([0, 0.05, 0.05, 2.0]), ub=np.array([10000, 0.5, 4, 300]))
        bounds = Bounds(lb=np.array([0, 0.01, 0.01, 2.0]), ub=np.array([10000, 0.5, 4, 300]))
        # perform fitting
        start = time.time()
        print_timer = 0
        for training_data in range(n_signals):

            if calculate_sigma:
                SNR = 1 / np.std(signal[training_data, :][sampling_scheme == 0])
            if sigma_array is not None:
                SNR = 1 / sigma_array[training_data]

            # compute + save model parameters
            optimize_result = minimize(fun=cost_gradient_descent, x0=seed[training_data, :],
                                       bounds=bounds, args=[signal[training_data, :], SNR,
                                                            model_name, noise_type, sampling_scheme])

            labels_bestfit[training_data, :] = optimize_result.x
            loss_bestfit[training_data, :] = optimize_result.fun

            if alternate_seed is not None:
                optimize_result_alt = minimize(fun=cost_gradient_descent, x0=alternate_seed[training_data, :],
                                               bounds=bounds, args=[signal[training_data, :], SNR,
                                                                    model_name, noise_type, sampling_scheme])
                if optimize_result_alt.fun < loss_bestfit[training_data, :]:
                    labels_bestfit[training_data, :] = optimize_result_alt.x

            elapsed_seconds = time.time() - start

            if int(elapsed_seconds) > print_timer:
                sys.stdout.write('\r')
                sys.stdout.write('MLE progress: {:.2f}%, elapsed time: {}'.format(training_data / n_signals * 100,
                                                                                  str(datetime.timedelta(
                                                                                      seconds=elapsed_seconds)).split(
                                                                                      ".")[0]))
                sys.stdout.flush()
                print_timer = print_timer + 1
            elif training_data == n_signals - 1:
                sys.stdout.write('\r')
                sys.stdout.write('...MLE done!'.format(training_data / n_signals * 100))
                sys.stdout.flush()
        sys.stdout.write('\r')

    else:
        sys.exit("Implement other signal models here")
    return labels_bestfit


def cost_gradient_descent(parameters, args):
    """ Function to compute loss for gradient descent optimisation

        Inputs
        ------
        parameter_array : ndarray
            parameter values to compute loss for

        args : list
            contains arguments needed to compute cost:
                args[0] : noisy_signal
                args[1] : SNR
                args[2] : model_name
                args[3] : noise_type
                args[4] : sampling_scheme

        Outputs
        -------
        cost: float64
            cost to be minimised by iterative GD fitting procedure

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    # Extract variables from args
    # measured (noisy) signal
    measured_signal = args[0]
    # standard deviation
    sigma = 1 / args[1]
    # model name
    model_name = args[2]
    # noise type
    noise_type = args[3]
    # sampling scheme
    sampling_scheme = args[4]
    if model_name == 'ivim_segmented':
        Dslow = args[5]
        generative_parameters = [parameters[0], parameters[1], Dslow, parameters[2]]
    else:
        generative_parameters = parameters
    # define mapping from model name to generative functions
    generative_signal_handler = {
        'ivim': generate_IVIM,
        'ivim_segmented': generate_IVIM,
    }
    # define mapping from noise type to cost function
    objective_fn_handler = {
        'rician': rician_log_likelihood,
        'gaussian': gaussian_sse
    }
    # compute noisefree synthetic signal associated with parameter estimates
    synth_signal = generative_signal_handler[model_name](generative_parameters, sampling_scheme)
    # compute cost associated with this predicted signal
    cost = objective_fn_handler[noise_type](synth_signal=synth_signal, meas_signal=measured_signal, sigma=sigma)
    return cost


def test_performance_wrapper(network_type, model_name, test_data, sampling_distribution,
                             script_dir, noise_type, dataset_size, SNR,
                             network_arch, network=None, mle_flag=False, mle_data=None):
    """ Function to evaluate fitting performance on test data + save results to disk

        Inputs
        ------
        network_type : string
            network name

        model_name : string
            name of signal model being fit

        test_data : method_classes.testDataset
            dataset to test network on

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        script_dir : string
            base directory of main script

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        dataset_size : int
            total number of signals generated for training and validation

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        network_arch : string
            network architecture, as defined in method_functions.train_general_network

        network : optional, method_classes.trainedNet
            trained network object, provided if testing ANN performance

        mle_flag : optional, bool
            if True, evaluate performance of conventional MLE rather than supplied network

        mle_data : optional, method_classes.testResults
            if mle_flag == True, provides conventional MLE model fit

        Outputs
        -------
        test_results : method_classes.testResults
            fitting performance on test data

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    if mle_flag:
        best_network = None
    else:
        best_network = network.best_network

    test_results = test_network(model_name=model_name,
                                network=best_network,
                                test_data=test_data.test_data_noisy_flat,
                                n_sampling=test_data.n_sampling,
                                n_repeats=test_data.n_repeats,
                                test_label=test_data.test_label,
                                mle_flag=mle_flag, mle_data=mle_data)

    # save test results
    temp_file = open(os.path.join(script_dir,
                                  'results/{}/{}/{}/{}/test_{}_{}_{}.pkl'.format(
                                      network_arch,
                                      model_name,
                                      noise_type,
                                      sampling_distribution,
                                      network_type,
                                      dataset_size,
                                      SNR)), 'wb')
    pickle.dump(test_results, temp_file)
    temp_file.close()

    return test_results


def generate_IVIM_tensor(label, sampling_scheme):
    """ Function to generate IVIM signal in differentiable tensor form

        Inputs
        ------
        label : ndarray
            groundtruth generative parameters

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme
            (independent variable values)

        Outputs
        -------
        signal : tensor
            noisefree signal

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    # read model parameters from tensor label object
    s0 = label[:, 0].unsqueeze(1)
    f = label[:, 1].unsqueeze(1)
    dslow = label[:, 2].unsqueeze(1)
    dfast = label[:, 3].unsqueeze(1)
    # generate tensor signal from model parameters
    signal = s0 * (
            f * torch.exp(-sampling_scheme * (dslow + dfast)) + (1 - f) * torch.exp(
        -sampling_scheme * dslow))
    return signal


def gaussian_sse(synth_signal, meas_signal, sigma):
    """ Function to calculate gaussian log likelihood, i.e. sum of squared errors

        Inputs
        ------
        synth_signal : ndarray
            array of noise-free signals synthesised from a generative model and proposed model parameters

        meas_signal : numpy array
            array of noisy experimental data

        sigma : float
            estimated standard deviation of gaussian distribution

        Outputs
        -------
        sse : float
            sum of squared errors between measured and synthesised signals

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    # compute sum of squared errors betweeen synthetic and measured signal
    sse = np.sum(np.square(synth_signal - meas_signal))
    return sse


def rician_log_likelihood(synth_signal, meas_signal, sigma):
    """ Function to calculate -1 * Rician log likelihood
        Obtained by taking the natural logarithm of the Rician PDF:
            P(meas_signal) =    meas_signal/sigma^2 *
                                modified_bessel_function_first_kind_zero_order(synth_signal * meas_signal/sigma^2) *
                                exp(-(synth_signal^2 + meas_signal^2)/(2 * sigma^2))
        and summing over all measurements.

        See Appendix of https://onlinelibrary.wiley.com/doi/10.1002/mrm.21646 for more info.

        Inputs
        ------
        synth_signal : ndarray
            array of of noise-free signals synthesised from a generative model and proposed model parameters

        meas_signal : numpy array
            array of noisy experimental data

        sigma : float
            estimated standard deviation of gaussian distribution underlying rician noise

        Outputs
        -------
        -log_likelihood : float
            Rician loglikelihood of meas_signal given synth_signal, sigma

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    # compute (synth_signal^2 + meas_signal^2)/(2*sigma^2)
    sum_square_signal_norm = (np.square(synth_signal) + np.square(meas_signal)) / (2 * np.square(sigma))
    # compute (synth_signal^2 + meas_signal^2)/(2*sigma^2)
    cross_term_norm = np.multiply(synth_signal, meas_signal) / (np.square(sigma))
    # natural log of modified Bessel function of the first kind of zeroth order
    # for x > 700, approximate log(besselio) directly as besselio outside floating point range
    index_approx = cross_term_norm > 700
    log_bessel_i0 = np.zeros_like(cross_term_norm)
    # approximation
    log_bessel_i0[index_approx] = cross_term_norm[index_approx] - np.log(
        2 * np.pi * cross_term_norm[index_approx]) / 2  # + np.reciprocal(cross_term_norm[index_approx] * 8)
    # exact calculation
    log_bessel_i0[~index_approx] = np.log(special.iv(0, cross_term_norm[~index_approx]))
    # compute log likelihood for each sampling point
    log_likelihood_array = -2 * np.log(sigma) - sum_square_signal_norm + np.log(meas_signal) + log_bessel_i0
    # sum over all sample points to obtain overall log likelihood
    log_likelihood = np.sum(log_likelihood_array)
    # return negative log likelihood (cost to minimise)
    return -log_likelihood


def train_general_network(script_dir, model_name, sampling_scheme, sampling_distribution, network_type, architecture,
                          n_nets, SNR, dataset_size, training_split, validation_split,
                          training_loader, validation_loader, method_name, validation_condition,
                          normalisation_factor, network_arch, noise_type, criterion, ignore_validation=False,
                          transfer_learning=False, state_dictionary=None):
    """ Function to train a DNN

        Inputs
        ------
        script_dir : string
            base directory of main script

        model_name : string
            name of signal model being fit

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme (independent
            variable values)

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        dataset_size : int
            total number of signals generated for training and validation

        training_split : optional, float
            proportion of dataset_size allocated to training

        validation_split : optional, float
            proportion of dataset_size allocated to validation

        training_loader: torch.utils.data.DataLoader
            pytorch dataloder for training data

        validation_loader: torch.utils.data.DataLoader
            pytorch dataloder for validation data

        method_name : string
            name of network training method

        validation_condition : int
            used to select a network with val_loss > train_loss / validation_condition  to avoid
            spuriously low validation loss when training with small number of samples

        normalisation_factor : ndarray
            defined in method_functions.get_parameter_scaling; relative weighting of each signal model
            parameter during supervised training; larger scaling results in lower weighting

        network_arch : string
            network architecture, see local variable architecture_dictionary

        noise_type : string
            name of noise type

        criterion : torch.nn loss criterion
            defines loss function used when training

        ignore_validation=False : optional, bool
            if True, ignores validation_condition

        transfer_learning=False : optional, bool
            if True, sets initial network weights to state_dictionary

        state_dictionary=None : optional, pytorch state_dict
            initial network weights

        Outputs
        -------
        trainedNetObj : method_classes.trainedNet
            object containing trained network(s)

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # set rng seeds
    torch.manual_seed(0)
    random.seed(0)
    # define mapping from network type to training method
    network_dictionary = {
        'supervised_parameters': train_net_super_params,
        'selfsupervised': train_net_selfsupervised,
        'hybrid': train_net_hybrid
    }

    # define mapping from network architecture to network class
    architecture_dictionary = {
        'NN_narrow': method_classes.netNarrow,
        'NN_narrow_mae': method_classes.netNarrow,
        'NN_narrow_extended': method_classes.netNarrow
    }
    # generate n_nets network objects
    network_object = []

    # if initialising network weights from another network, no need to train multiple times
    if transfer_learning:
        n_nets = 1
    # generate network objects to train
    for i in range(n_nets):
        network_object.append(architecture_dictionary[architecture](sampling_scheme=sampling_scheme,
                                                                    n_params=get_n_free_parameters(model_name),
                                                                    model_name=model_name))
    # initialise network weights from another network
    if transfer_learning:
        network_object[0].load_state_dict(state_dictionary)

    # set pytorch optimizer
    optimizer = [optim.Adam(a.parameters(), lr=0.001) for a in network_object]
    # initialise network loss tracker (training and validation)
    training_loss = [[] for a in network_object]
    validation_loss = [[] for a in network_object]
    # initialise # of epochs tracker
    last_epoch_tracker = [[] for a in network_object]
    # initialise # loss tracker for each minibatch
    batch_loss = [[] for a in network_object]
    # initialise network tracker for each epoch
    network_tracker = [[] for a in network_object]

    # start multiprocessing pool
    p = Pool()
    # train networks
    for network_index, worker_output in enumerate(p.starmap(network_dictionary[network_type],
                                                            zip(network_object,
                                                                optimizer,
                                                                itertools.repeat(criterion),
                                                                itertools.repeat(training_loader),
                                                                itertools.repeat(validation_loader),
                                                                range(n_nets),
                                                                itertools.repeat(method_name),
                                                                itertools.repeat(normalisation_factor)
                                                                ))):
        network_object[network_index] = worker_output[0]
        training_loss[network_index] = worker_output[1]
        validation_loss[network_index] = worker_output[2]
        last_epoch_tracker[network_index] = worker_output[3]
        batch_loss[network_index] = worker_output[4]
        network_tracker[network_index] = worker_output[5]

    # order networks by training or validation loss
    loss_tracker_ordering = np.zeros([n_nets])
    conditional_loss_tracker = np.zeros([n_nets])
    # if only one network trained, choose it as best by default
    if transfer_learning:
        sorted_networks_idx = [0]
        network_idx = sorted_networks_idx[0]
        best_network = network_object[network_idx]
    # if validation ignored, simply order network by training loss and pick best
    elif ignore_validation:
        # iterate over training loss history
        for i, train_loss in enumerate(training_loss):
            # NN_wide determine training loss associated with final network state
            loss_tracker_ordering[i] = train_loss[last_epoch_tracker[i]]
        # sort networks by their training loss
        sorted_networks_idx = np.argsort(loss_tracker_ordering)
        # determine best network
        network_idx = sorted_networks_idx[0]
        best_network = network_object[network_idx]
    # order networks by validation loss, ensuring val_loss > train_loss / validation_condition (avoid spuriously low
    # validation loss when training with small number of samples)
    else:
        # iterate over validation loss history
        for i, val_loss in enumerate(validation_loss):
            # determine validation loss associated with final network state
            loss_tracker_ordering[i] = val_loss[last_epoch_tracker[i]]
            # sort networks by their validation loss
        sorted_networks_idx = np.argsort(loss_tracker_ordering)
        # determine best network by ensuring val_loss > train_loss / n
        if val_loss[last_epoch_tracker[i]] / validation_split > training_loss[i][
            last_epoch_tracker[i]] / training_split / validation_condition:
            conditional_loss_tracker[i] = val_loss[last_epoch_tracker[i]]
        # determine index of best network, ignoring networks where conditional_loss_tracker == 0
        network_idx = np.argmin(ma.masked_where(conditional_loss_tracker == 0, conditional_loss_tracker))
        # determine best network
        best_network = network_object[network_idx]
    # save and return trainedNet object
    trainedNetObj = method_classes.trainedNet(n_nets=n_nets,
                                              architecture=architecture,
                                              network_type=network_type,
                                              best_network=best_network,
                                              network_object=network_object,
                                              best_network_idx=network_idx,
                                              training_loss=training_loss,
                                              validation_loss=validation_loss,
                                              last_epoch_tracker=last_epoch_tracker,
                                              sorted_networks_idx=sorted_networks_idx,
                                              batch_loss=batch_loss,
                                              network_tracker=network_tracker)

    # save trained networks
    temp_file = open(
        os.path.join(script_dir,
                     'models/{}/{}/{}/{}/{}_{}_{}.pkl'.format(network_arch,
                                                              model_name,
                                                              noise_type,
                                                              sampling_distribution,
                                                              method_name,
                                                              dataset_size,
                                                              SNR)), 'wb')
    pickle.dump(trainedNetObj, temp_file)
    temp_file.close()

    return trainedNetObj


def train_net_super_params(netobj, optimizer, criterion, train_dataloader, val_dataloader, net_counter=1,
                           network_name='supervised params', normalisation_factor=None):
    """ Function to train a supervised DNN

        Inputs
        ------
        netobj : network object, e.g. method_classes.netNarrow
            network to be trained

        optimizer: torch.optim optimizer
            pytorch optimization algorithm

        criterion : torch.nn loss criterion
            defines loss function used when training

        train_dataloader: torch.utils.data.DataLoader
            pytorch dataloder for training data

        val_dataloader: torch.utils.data.DataLoader
            pytorch dataloder for validation data

        net_counter=1 : optional, int
            network index being trained

        network_name='supervised params' : optional, string
            name of network being trained

        normalisation_factor=None : optional, ndarray
            defined in method_functions.get_parameter_scaling; relative weighting of each signal
            model parameter during supervised training; larger scaling results in lower weighting

        Outputs
        -------
        netobj :  network object, e.g. method_classes.netNarrow
            trained network

        train_loss_tracking : ndarray
            tracks training loss for each epoch

        val_loss_tracking : ndarray
            tracks validation loss for each epoch

        epoch : int
            final epoch reached during training

        batch_loss :  ndarray
            tracks training loss for each minibatch

        model_tracking : list
            tracks network state evolution during training

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    if normalisation_factor is None:
        normalisation_factor = [1, 1, 1, 1]
    # initialise overall cost tracker
    best_cost = 1e16
    # initialise bad epoch tracker
    bad_epochs = 0
    # max number of epochs
    n_epochs = 150000 // len(train_dataloader)
    # number of bad epochs allowed before giving up
    patience = n_epochs // 5
    # initialise loss tracker for each network step
    train_loss_tracking = np.zeros(shape=(n_epochs, 1))
    val_loss_tracking = np.zeros(shape=(n_epochs, 1))
    # train network
    # for epoch in range(1000):
    batch_counter = 0
    batch_loss = np.zeros(shape=(n_epochs * len(train_dataloader), 1))
    model_tracking = []
    final_model = netobj.state_dict()
    for epoch in range(n_epochs):
        # set training state
        netobj.train()
        # zero training and validation losses
        train_running_loss = 0.
        val_running_loss = 0.
        # iterate over batches in training dataloader
        for i, (batch_data, batch_label) in enumerate(train_dataloader):
            # extract batch size
            batch_size = batch_label.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # obtain network prediction
            _, batch_predict = netobj(batch_data)
            # compute loss associated with network prediction
            train_loss = criterion(torch.div(batch_predict, torch.tensor(normalisation_factor).repeat(batch_size, 1)),
                                   torch.div(batch_label, torch.tensor(normalisation_factor).repeat(batch_size, 1)))
            # backpropagate loss, accumulate gradient
            train_loss.backward()
            # update parameters based on computed gradients
            optimizer.step()
            # add the batch loss to the epoch loss tracker
            train_running_loss += train_loss.item()
            # store minibatch loss
            batch_loss[batch_counter, 0] = train_loss.item()
            # update number of batch updates
            batch_counter += 1
        # set network to evaluation state to compute validation loss
        netobj.eval()
        with torch.no_grad():
            # iterate over batches in validation dataloader
            for i, (batch_data, batch_label) in enumerate(val_dataloader):
                # extract batch size
                batch_size = batch_label.shape[0]
                # obtain network prediction
                _, batch_predict = netobj(batch_data)
                # compute loss associated with network prediction
                val_loss = criterion(
                    torch.div(batch_predict, torch.tensor(normalisation_factor).repeat(batch_size, 1)),
                    torch.div(batch_label, torch.tensor(normalisation_factor).repeat(batch_size, 1)))
                # add the batch loss to the epoch loss tracker
                val_running_loss += val_loss.item()
        # early stopping
        if train_running_loss < best_cost:
            print("############# Saving good model #############")
            print(
                "Network type: {}; Network number: {}; Current epoch: {}".format(network_name, net_counter + 1, epoch))
            print("Loss: {}".format(train_running_loss))
            print("---------------------------------------------")
            # store best model
            # store loss associated with all batches for current epoch
            train_loss_tracking[epoch, 0] = train_running_loss
            val_loss_tracking[epoch, 0] = val_running_loss

            final_model = netobj.state_dict()
            best_cost = train_running_loss
            bad_epochs = 0
            model_tracking.append((epoch, netobj))
        else:
            bad_epochs = bad_epochs + 1
            train_loss_tracking[epoch, 0] = train_loss_tracking[epoch - 1, 0]
            val_loss_tracking[epoch, 0] = val_loss_tracking[epoch - 1, 0]
            if bad_epochs == patience:
                print("Network type: {}; Network number: {}; Done, best loss: {}".format(network_name, net_counter + 1,
                                                                                         best_cost))
                print("-----------------------------------------")
                break

    # Restore best model
    netobj.load_state_dict(final_model)
    return netobj, train_loss_tracking, val_loss_tracking, epoch, batch_loss, model_tracking


def train_net_selfsupervised(netobj, optimizer, criterion, train_dataloader, val_dataloader, net_counter=1,
                             network_name='selfsupervised params', normalisation_factor=None):
    """ Function to train a selfsupervised DNN

        Inputs
        ------
        netobj : network object, e.g. method_classes.netNarrow
            network to be trained

        optimizer: torch.optim optimizer
            pytorch optimization algorithm

        criterion : torch.nn loss criterion
            defines loss function used when training

        train_dataloader: torch.utils.data.DataLoader
            pytorch dataloder for training data

        val_dataloader: torch.utils.data.DataLoader
            pytorch dataloder for validation data

        net_counter=1 : optional, int
            network index being trained

        network_name='selfsupervised params' : optional, string
            name of network being trained

        normalisation_factor=None : optional, ndarray
            defined in method_functions.get_parameter_scaling; relative weighting of each signal
            model parameter during supervised training; larger scaling results in lower weighting

        Outputs
        -------
        netobj :  network object, e.g. method_classes.netNarrow
            trained network

        train_loss_tracking : ndarray
            tracks training loss for each epoch

        val_loss_tracking : ndarray
            tracks validation loss for each epoch

        epoch : int
            final epoch reached during training

        batch_loss :  ndarray
            tracks training loss for each minibatch

        model_tracking : list
            tracks network state evolution during training

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # initialise overall cost tracker
    best_cost = 1e16
    # initialise bad epoch tracker
    bad_epochs = 0
    # max number of epochs
    n_epochs = 150000 // len(train_dataloader)
    # number of bad epochs allowed before giving up
    patience = n_epochs // 5
    # initialise loss tracker for each network step
    train_loss_tracking = np.zeros(shape=(n_epochs, 1))
    val_loss_tracking = np.zeros(shape=(n_epochs, 1))
    # train network
    batch_counter = 0
    batch_loss = np.zeros(shape=(n_epochs * len(train_dataloader), 1))
    model_tracking = []
    final_model = netobj.state_dict()
    for epoch in range(n_epochs):
        # set training state
        netobj.train()
        # zero training and validation losses
        train_running_loss = 0.
        val_running_loss = 0.
        # iterate over batches in training dataloader
        for i, (batch_data, _) in enumerate(train_dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()
            # obtain network prediction
            batch_predict_signal, _ = netobj(batch_data)
            # compute loss associated with network prediction
            train_loss = criterion(batch_predict_signal, batch_data)
            # backpropagate loss, accumulate gradient
            train_loss.backward()
            # update parameters based on computed gradients
            optimizer.step()
            # add the batch loss to the epoch loss tracker
            train_running_loss += train_loss.item()
            # store minibatch loss
            batch_loss[batch_counter, 0] = train_loss.item()
            # update number of batch updates
            batch_counter += 1
        # set network to evaluation state to compute validation loss
        netobj.eval()
        with torch.no_grad():
            # iterate over batches in validation dataloader
            for i, (batch_data, _) in enumerate(val_dataloader):
                # obtain network prediction
                batch_predict_signal, _ = netobj(batch_data)
                # compute loss associated with network prediction
                val_loss = criterion(batch_predict_signal, batch_data)
                # add the batch loss to the epoch loss tracker
                val_running_loss += val_loss.item()
        # early stopping
        if train_running_loss < best_cost:
            print("############# Saving good model #############")
            print(
                "Network type: {}; Network number: {}; Current epoch: {}".format(network_name, net_counter + 1, epoch))
            print("Loss: {}".format(train_running_loss))
            print("---------------------------------------------")
            # store best model
            # store loss associated with all batches for current epoch
            train_loss_tracking[epoch, 0] = train_running_loss
            val_loss_tracking[epoch, 0] = val_running_loss

            final_model = netobj.state_dict()
            best_cost = train_running_loss
            bad_epochs = 0
            model_tracking.append((epoch, netobj))
        else:
            bad_epochs = bad_epochs + 1
            train_loss_tracking[epoch, 0] = train_loss_tracking[epoch - 1, 0]
            val_loss_tracking[epoch, 0] = val_loss_tracking[epoch - 1, 0]
            if bad_epochs == patience:
                print("Done, best loss: {}".format(best_cost))
                print("-----------------------------------------")
                break

    # Restore best model
    netobj.load_state_dict(final_model)
    return netobj, train_loss_tracking, val_loss_tracking, epoch, batch_loss, model_tracking


def train_net_hybrid(netobj, optimizer, criterion, train_dataloader, val_dataloader, net_counter=1,
                     network_name='hybrid', normalisation_factor=None):
    """ Function to train a hybrid DNN

        Inputs
        ------
        netobj : network object, e.g. method_classes.netNarrow
            network to be trained

        optimizer: torch.optim optimizer
            pytorch optimization algorithm

        criterion : torch.nn loss criterion
            defines loss function used when training

        train_dataloader: torch.utils.data.DataLoader
            pytorch dataloader for training data

        val_dataloader: torch.utils.data.DataLoader
            pytorch dataloader for validation data

        net_counter=1 : optional, int
            network index being trained

        network_name='hybrid params' : optional, string
            name of network being trained

        normalisation_factor=None : optional, ndarray
            defined in method_functions.get_parameter_scaling; relative weighting of each signal
            model parameter during supervised training; larger scaling results in lower weighting

        Outputs
        -------
        netobj :  network object, e.g. method_classes.netNarrow
            trained network

        train_loss_tracking : ndarray
            tracks training loss for each epoch

        val_loss_tracking : ndarray
            tracks validation loss for each epoch

        epoch : int
            final epoch reached during training

        batch_loss :  ndarray
            tracks training loss for each minibatch

        model_tracking : list
            tracks network state evolution during training

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    if normalisation_factor is None:
        normalisation_factor = [1, 1, 1, 1]
    # initialise overall cost tracker
    best_cost = 1e16
    # initialise bad epoch tracker
    bad_epochs = 0
    # max number of epochs
    n_epochs = 150000 // len(train_dataloader)
    # number of bad epochs allowed before giving up
    patience = n_epochs // 5
    # initialise loss tracker for each network step
    train_loss_tracking = np.zeros(shape=(n_epochs, 1))
    val_loss_tracking = np.zeros(shape=(n_epochs, 1))
    # train network
    # for epoch in range(1000):
    batch_counter = 0
    batch_loss = np.zeros(shape=(n_epochs * len(train_dataloader), 1))
    model_tracking = []
    final_model = netobj.state_dict()
    for epoch in range(n_epochs):
        # set training state
        netobj.train()
        # zero training and validation losses
        train_running_loss = 0.
        val_running_loss = 0.
        # iterate over batches in training dataloader
        for i, (batch_data, batch_label) in enumerate(train_dataloader):
            # extract batch size
            batch_size = batch_label.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # obtain network prediction
            _, batch_predict = netobj(batch_data)
            # compute loss associated with network prediction
            train_loss = 0.5 * criterion(
                torch.div(batch_predict, torch.tensor(normalisation_factor).repeat(batch_size, 1)),
                torch.div(batch_label[:, :, 0], torch.tensor(normalisation_factor).repeat(batch_size, 1))) + \
                         0.5 * criterion(
                torch.div(batch_predict, torch.tensor(normalisation_factor).repeat(batch_size, 1)),
                torch.div(batch_label[:, :, 1], torch.tensor(normalisation_factor).repeat(batch_size, 1)))
            # backpropagate loss, accumulate gradient
            train_loss.backward()
            # update parameters based on computed gradients
            optimizer.step()
            # add the batch loss to the epoch loss tracker
            train_running_loss += train_loss.item()
            # store minibatch loss
            batch_loss[batch_counter, 0] = train_loss.item()
            # update number of batch updates
            batch_counter += 1
        # set network to evaluation state to compute validation loss
        netobj.eval()
        with torch.no_grad():
            # iterate over batches in validation dataloader
            for i, (batch_data, batch_label) in enumerate(val_dataloader):
                # extract batch size
                batch_size = batch_label.shape[0]
                # obtain network prediction
                _, batch_predict = netobj(batch_data)
                # compute loss associated with network prediction
                val_loss = 0.5 * criterion(
                    torch.div(batch_predict, torch.tensor(normalisation_factor).repeat(batch_size, 1)),
                    torch.div(batch_label[:, :, 0], torch.tensor(normalisation_factor).repeat(batch_size, 1))) + \
                           0.5 * criterion(
                    torch.div(batch_predict, torch.tensor(normalisation_factor).repeat(batch_size, 1)),
                    torch.div(batch_label[:, :, 1], torch.tensor(normalisation_factor).repeat(batch_size, 1)))
                # add the batch loss to the epoch loss tracker
                val_running_loss += val_loss.item()
        # early stopping
        if train_running_loss < best_cost:
            print("############# Saving good model #############")
            print(
                "Network type: {}; Network number: {}; Current epoch: {}".format(network_name, net_counter + 1, epoch))
            print("Loss: {}".format(train_running_loss))
            print("---------------------------------------------")
            # store best model
            # store loss associated with all batches for current epoch
            train_loss_tracking[epoch, 0] = train_running_loss
            val_loss_tracking[epoch, 0] = val_running_loss

            final_model = netobj.state_dict()
            best_cost = train_running_loss
            bad_epochs = 0
            model_tracking.append((epoch, netobj))
        else:
            bad_epochs = bad_epochs + 1
            train_loss_tracking[epoch, 0] = train_loss_tracking[epoch - 1, 0]
            val_loss_tracking[epoch, 0] = val_loss_tracking[epoch - 1, 0]
            if bad_epochs == patience:
                print("Network type: {}; Network number: {}; Done, best loss: {}".format(network_name, net_counter + 1,
                                                                                         best_cost))
                print("-----------------------------------------")
                break

    # Restore best model
    netobj.load_state_dict(final_model)
    return netobj, train_loss_tracking, val_loss_tracking, epoch, batch_loss, model_tracking


def create_test_data(script_dir, model_name, sampling_distribution, noise_type, SNR, dataset_size, training_data,
                     n_repeats, n_sampling, extent_scaling):
    """ Function to create test dataset

        Inputs
        ------
        script_dir : string
            base directory of main script

        model_name : string
            name of signal model being fit

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        dataset_size : int
            total number of signals generated for training and validation

        training_data : method_classes.Dataset
            object containing training data and dataloaders

        n_sampling : ndarray
            number of sampling points in each model parameter dimension

        n_repeats : int
            number of repetitions at each sampling point

        extent_scaling = float
            extent to which networks are tested outside the range they were trained on

        Outputs
        -------
        test_data : method_classes.testDataset
            object containing test data

        test_mle_groundtruth : method_classes.testResults
            object containing conventional MLE fits (seeded with groundtruth values) of the test data

        test_mle_mean : method_classes.testResults
            object containing conventional MLE fits (seeded with mean values) of the test data

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # extract training parameter range from training data
    parameter_range = training_data.parameter_range

    if model_name == 'ivim':

        # determine sampling points, uniformly distributed over parameter space
        theta_0_uniform = np.array([1.0])
        theta_1_uniform = np.linspace(
            parameter_range[1, 0] - (parameter_range[1, 1] - parameter_range[1, 0]) * extent_scaling,
            parameter_range[1, 1] + (parameter_range[1, 1] - parameter_range[1, 0]) * extent_scaling,
            n_sampling[1])
        theta_2_uniform = np.linspace(
            parameter_range[2, 0] - (parameter_range[2, 1] - parameter_range[2, 0]) * extent_scaling,
            parameter_range[2, 1] + (parameter_range[2, 1] - parameter_range[2, 0]) * extent_scaling,
            n_sampling[2])
        theta_3_uniform = np.linspace(
            parameter_range[3, 0] - (parameter_range[3, 1] - parameter_range[3, 0]) * extent_scaling,
            parameter_range[3, 1] + (parameter_range[3, 1] - parameter_range[3, 0]) * extent_scaling,
            n_sampling[3])

        # compute meshgrid from 1D thetas, and combine into a single numpy array
        label_uniform = np.transpose(
            np.stack((np.meshgrid(theta_0_uniform, theta_1_uniform, theta_2_uniform, theta_3_uniform, indexing='ij'))),
            (1, 2, 3, 4, 0))

        # compute MLE for uniformly sampled test data
        test_data, test_mle_groundtruth, test_mle_mean = \
            analyse_test_data(n_repeats=n_repeats,
                              n_sampling=n_sampling,
                              sampling_scheme=training_data.sampling_scheme,
                              model_name=training_data.model_name,
                              label=label_uniform,
                              noise_type=training_data.noise_type,
                              SNR=training_data.SNR,
                              extent_scaling=extent_scaling)
    else:
        sys.exit("Implement other signal models here")

    temp_file = open(os.path.join(script_dir,
                                  'data/test/{}/{}/{}/test_data_{}_{}.pkl'.format(model_name,
                                                                                  noise_type,
                                                                                  sampling_distribution,
                                                                                  dataset_size,
                                                                                  SNR)), 'wb')
    pickle.dump(test_data, temp_file)
    temp_file.close()

    temp_file = open(os.path.join(script_dir,
                                  'data/test/{}/{}/{}/test_MLE_groundtruth_{}_{}.pkl'.format(model_name,
                                                                                             noise_type,
                                                                                             sampling_distribution,
                                                                                             dataset_size,
                                                                                             SNR)), 'wb')
    pickle.dump(test_mle_groundtruth, temp_file)
    temp_file.close()

    temp_file = open(os.path.join(script_dir,
                                  'data/test/{}/{}/{}/test_MLE_mean_{}_{}.pkl'.format(model_name,
                                                                                      noise_type,
                                                                                      sampling_distribution,
                                                                                      dataset_size,
                                                                                      SNR)), 'wb')
    pickle.dump(test_mle_mean, temp_file)
    temp_file.close()

    return test_data, test_mle_groundtruth, test_mle_mean


def analyse_test_data(SNR, n_sampling, n_repeats, extent_scaling, sampling_scheme, model_name, label, noise_type):
    """ Function to analyse test dataset

        Inputs
        ------

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        n_sampling : ndarray
            number of sampling points in each model parameter dimension

        n_repeats : int
            number of repetitions at each sampling point

        extent_scaling = float
            extent to which networks are tested outside the range they were trained on

        sampling_scheme : ndarray
            provides signal sampling scheme (independent variable values)

        model_name : string
            name of signal model being fit

        label : ndarray
            groundtruth generative parameter associated with each test signal

        noise_type : string
            name of noise type


        Outputs
        -------
        test_data : method_classes.testDataset
            object containing test data

        mle_results_groundtruth : method_classes.testResults
            object containing conventional MLE fits (seeded with groundtruth values) of the test data

        mle_results_mean : method_classes.testResults
            object containing conventional MLE fits (seeded with mean values) of the test data

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    generative_dictionary = {
        'ivim': generate_IVIM
    }

    if model_name == 'ivim':

        # generate noisefree test data
        data_noisefree = np.zeros(
            (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats, len(sampling_scheme)))

        #  generate signal corresponding to each label
        for theta_0 in range(n_sampling[0]):
            for theta_1 in range(n_sampling[1]):
                for theta_2 in range(n_sampling[2]):
                    for theta_3 in range(n_sampling[3]):
                        for repeat in range(n_repeats):
                            temp_label = []
                            for i in range(label.shape[4]):
                                temp_label.append(label[theta_0, theta_1, theta_2, theta_3, i])

                            data_noisefree[theta_0, theta_1, theta_2, theta_3, repeat, :] = generative_dictionary[
                                model_name](
                                temp_label,
                                sampling_scheme)
        # reshape noisefree data
        data_noisefree_flat = np.reshape(data_noisefree,
                                         (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * n_repeats,
                                          len(sampling_scheme)))
        # reshape labels
        label_flat = matlib.repmat(
            np.reshape(label, (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3], 4)), n_repeats, 1)

        # add noise
        data_noisy_flat = add_noise(noise_type, SNR, data_noisefree_flat)
        # reshape noisy data
        data_noisy = np.reshape(data_noisy_flat,
                                (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats,
                                 len(sampling_scheme)))

        # compute MLE estimates using groundtruth seeds
        MLE_params_groundtruth = np.reshape(traditional_fitting(model_name=model_name,
                                                                signal=data_noisy_flat,
                                                                sampling_scheme=sampling_scheme,
                                                                labels_groundtruth=label_flat,
                                                                SNR=SNR,
                                                                noise_type=noise_type,
                                                                seed_mean=False),
                                            (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats, 4))

        # calculate mean MLE param
        MLE_params_groundtruth_mean = np.mean(MLE_params_groundtruth, axis=4)
        # calculate MLE standard deviation under noise
        MLE_params_groundtruth_std = np.std(MLE_params_groundtruth, axis=4)

        MLE_params_groundtruth_flat = np.reshape(np.transpose(MLE_params_groundtruth, (0, 1, 2, 3, 5, 4)),
                                                 (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4,
                                                  n_repeats))
        groundtruth_params_flat = matlib.repmat(
            np.reshape(label, (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4, 1)), 1, n_repeats)

        MLE_params_groundtruth_rmse = np.reshape(
            np.sqrt(np.mean(((MLE_params_groundtruth_flat - groundtruth_params_flat) ** 2), axis=1)),
            (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], 4))

        # compute MLE estimates using mean seeds
        MLE_params_mean = np.reshape(traditional_fitting(model_name=model_name,
                                                         signal=data_noisy_flat,
                                                         sampling_scheme=sampling_scheme,
                                                         labels_groundtruth=label_flat,
                                                         SNR=SNR,
                                                         noise_type=noise_type,
                                                         seed_mean=True),
                                     (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats, 4), )

        MLE_params_mean_mean = np.mean(MLE_params_mean, axis=4)
        MLE_params_mean_std = np.std(MLE_params_mean, axis=4)

        MLE_params_mean_flat = np.reshape(np.transpose(MLE_params_mean, (0, 1, 2, 3, 5, 4)),
                                          (
                                              n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4,
                                              n_repeats))

        groundtruth_params_flat = matlib.repmat(
            np.reshape(label, (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4, 1)), 1, n_repeats)

        MLE_params_mean_rmse = np.reshape(
            np.sqrt(np.mean(((MLE_params_mean_flat - groundtruth_params_flat) ** 2), axis=1)),
            (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], 4))

    mle_results_groundtruth = method_classes.testResults(network_object=None,
                                                         test_data=data_noisy,
                                                         param_predictions=MLE_params_groundtruth,
                                                         param_predictions_mean=MLE_params_groundtruth_mean,
                                                         param_predictions_std=MLE_params_groundtruth_std,
                                                         param_predictions_rmse=MLE_params_groundtruth_rmse)

    mle_results_mean = method_classes.testResults(network_object=None,
                                                  test_data=data_noisy,
                                                  param_predictions=MLE_params_mean,
                                                  param_predictions_mean=MLE_params_mean_mean,
                                                  param_predictions_std=MLE_params_mean_std,
                                                  param_predictions_rmse=MLE_params_mean_rmse)

    test_data = method_classes.testDataset(test_data=data_noisefree,
                                           test_data_flat=data_noisefree_flat,
                                           test_label=label,
                                           test_label_flat=None,
                                           test_data_noisy=data_noisy,
                                           test_data_noisy_flat=data_noisy_flat,
                                           n_repeats=n_repeats,
                                           n_sampling=n_sampling,
                                           extent_scaling=extent_scaling)

    return test_data, mle_results_groundtruth, mle_results_mean


def test_network(model_name, test_data, test_label, n_sampling, n_repeats,
                 mle_flag=False, mle_data=None, network=None, clinical_flag=False):
    """ Function to test network performance

        Inputs
        ------

        model_name : string
            name of signal model being fit

        test_data : method_classes.testDataset
            object containing test data

        test_label : ndarray
            groundtruth generative parameter associated with each test signal

        n_sampling : ndarray
            number of sampling points in each model parameter dimension

        n_repeats : int
            number of repetitions at each sampling point

        mle_flag=False : optional, bool
            set to True if assessing conventional MLE performance

        mle_data=None : optional, method_classes.testResults
            provides MLE estimates if mle_flag=True

        network=None : optional, network object, e.g. method_classes.netNarrow
            network to be tested, if mle_flag=False

        clinical_flag=False : optional, boolean
            set to True if assessing clinical data

        Outputs
        -------
        mle_results : method_classes.testResults
            object containing parameter fits of the test data

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    # if assessing clinically-derived data, normalise by s(b=0)
    if clinical_flag:
        test_data_norm = np.divide(test_data, np.tile(test_data[:, 0], (test_data.shape[1], 1)).T)
    else:
        test_data_norm = test_data

    # if assessing conventional fit, extract parameters from MLE data
    if mle_flag:
        fitted_parameters = mle_data.param_predictions
    else:
        # set network to evaluation mode
        network.eval()
        # obtain network predictions
        with torch.no_grad():
            fitted_signal, fitted_parameters = network(torch.from_numpy(test_data_norm.astype(np.float32)))

    if model_name == 'ivim':

        if mle_flag:
            pass
        elif clinical_flag:
            fitted_parameters[:, 0] = np.multiply(fitted_parameters[:, 0], test_data[:, 0])
            fitted_parameters = np.reshape(fitted_parameters.numpy(),
                                           (n_sampling, n_repeats, 4))
        else:
            # reshaped best-fit parameters
            fitted_parameters = np.reshape(fitted_parameters.numpy(),
                                           (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3],
                                            n_repeats, 4))

        if clinical_flag:
            #  mean best-fit parameters under noise
            fitted_parameters_mean = np.mean(fitted_parameters, axis=1)
            # standard deviation of best-fit parameters under noise
            fitted_parameters_std = np.std(fitted_parameters, axis=1)
        else:
            #  mean best-fit parameters under noise
            fitted_parameters_mean = np.mean(fitted_parameters, axis=4)
            # standard deviation of best-fit parameters under noise
            fitted_parameters_std = np.std(fitted_parameters, axis=4)

        # reshaped groundtruth parameters, used to compute rmse
        if clinical_flag:
            test_label_expanded = np.swapaxes(np.reshape(matlib.repmat(
                np.reshape(test_label, (n_sampling * 4, 1)), 1, n_repeats), (n_sampling, 4, n_repeats)), 1, 2)
        else:
            test_label_flat = matlib.repmat(
                np.reshape(test_label, (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4, 1)), 1,
                n_repeats)
            test_label_expanded = np.swapaxes(
                np.reshape(test_label_flat, (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], 4, n_repeats)),
                4, 5)

        # compute rmse of best-fit parameters
        if clinical_flag:
            fitted_parameters_rmse = np.sqrt(np.mean((fitted_parameters - test_label_expanded) ** 2, axis=1))
        else:
            fitted_parameters_rmse = np.sqrt(np.mean((fitted_parameters - test_label_expanded) ** 2, axis=4))
    else:
        sys.exit("Implement other signal models here")

    return method_classes.testResults(network_object=network,
                                      test_data=test_data,
                                      param_predictions=fitted_parameters,
                                      param_predictions_mean=fitted_parameters_mean,
                                      param_predictions_std=fitted_parameters_std,
                                      param_predictions_rmse=fitted_parameters_rmse)


def create_test_data_clinical(label, n_repeats, model_name, script_dir, noise_type, sampling_scheme,
                              sampling_distribution, sigma):
    """ Function to create synthetic test dataset which matches a supplied clinical data distribution

        Inputs
        ------

        label : ndarray
            clinical groundtruth parameters to use as generative values for synthetic dataset

        n_repeats : int
            number of repetitions at each sampling point

        model_name : string
            name of signal model being fit

        script_dir : string
            base directory of main script

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        sampling_scheme : ndarray
            provides signal sampling scheme (independent variable values)

        sampling_distribution : string
            name of sampling distribution

        sigma : ndarray
            standard deviation of each signal to be generated


        Outputs
        -------
        test_data : method_classes.testDataset
            object containing test data

        test_mle_mean : method_classes.testResults
            object containing conventional MLE fits (seeded with fixed values) of the test data

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    generative_dictionary = {
        'ivim': generate_IVIM
    }

    if model_name == 'ivim':

        # generate noisefree test data
        data_noisefree = np.zeros((label.shape[0], n_repeats, len(sampling_scheme)))

        #  generate signal corresponding to each label
        for label_idx in range(label.shape[0]):
            for repeat_idx in range(n_repeats):
                data_noisefree[label_idx, repeat_idx, :] = generative_dictionary[
                    model_name](
                    label[label_idx],
                    sampling_scheme)
        # reshape noisefree data
        data_noisefree_flat = np.reshape(data_noisefree,
                                         (label.shape[0] * n_repeats,
                                          len(sampling_scheme)))

        # add noise
        data_noisy = add_noise(noise_type=noise_type, SNR=np.divide(1, sigma), signal=data_noisefree,
                               clinical_flag=True)
        data_noisy_flat = np.reshape(data_noisy, (data_noisy.shape[0] * data_noisy.shape[1], data_noisy.shape[2]))

        sigma_repeat = np.reshape(np.matlib.repmat(sigma, 1, n_repeats), sigma.shape[0] * n_repeats)

        print('...done!')
        print('Calculating corresponding MLE estimates using mean seeds...')

        fitting_seed = np.zeros((label.shape[0] * n_repeats, 4))

        fitting_seed[:, 0] = data_noisy_flat[:, 0]
        fitting_seed[:, 1] = 0.3
        fitting_seed[:, 2] = 1.5
        fitting_seed[:, 3] = 50

        # compute MLE estimates using mean seeds
        MLE_params_mean = np.reshape(traditional_fitting(model_name=model_name,
                                                         signal=data_noisy_flat,
                                                         sampling_scheme=sampling_scheme,
                                                         labels_groundtruth=fitting_seed,
                                                         SNR=None,
                                                         noise_type=noise_type,
                                                         seed_mean=False,
                                                         calculate_sigma=False,
                                                         sigma_array=sigma_repeat),
                                     (label.shape[0], n_repeats, 4))
        print('...done!')

        # calculate mean MLE param
        MLE_params_mean_mean = np.mean(MLE_params_mean, axis=1)
        # calculate MLE standard deviation under noise
        MLE_params_mean_std = np.std(MLE_params_mean, axis=1)

        MLE_params_mean_flat = np.reshape(np.transpose(MLE_params_mean, (0, 2, 1)), (label.shape[0] * 4,
                                                                                     n_repeats))

        groundtruth_params_flat = matlib.repmat(
            np.reshape(label, (label.shape[0] * 4, 1)), 1, n_repeats)

        MLE_params_mean_rmse = np.reshape(np.sqrt(
            np.mean(((MLE_params_mean_flat - groundtruth_params_flat) ** 2), axis=1)),
            (label.shape[0], 4))
        test_mle_mean = method_classes.testResults(network_object=None,
                                                   test_data=data_noisy,
                                                   param_predictions=MLE_params_mean,
                                                   param_predictions_mean=MLE_params_mean_mean,
                                                   param_predictions_std=MLE_params_mean_std,
                                                   param_predictions_rmse=MLE_params_mean_rmse)

        test_data = method_classes.testDataset(test_data=data_noisefree,
                                               test_data_flat=data_noisefree_flat,
                                               test_label=label,
                                               test_label_flat=None,
                                               test_data_noisy=data_noisy,
                                               test_data_noisy_flat=data_noisy_flat,
                                               n_repeats=n_repeats,
                                               n_sampling=None,
                                               extent_scaling=None)

    else:
        sys.exit("Implement other signal models here")

    temp_file = open(os.path.join(script_dir,
                                  'data/test/{}/{}/{}/test_clinical_synth_dataset.pkl'.format(model_name,
                                                                                              noise_type,
                                                                                              sampling_distribution)),
                     'wb')
    pickle.dump(test_data, temp_file)
    temp_file.close()

    temp_file = open(os.path.join(script_dir,
                                  'data/test/{}/{}/{}/test_clinical_synth_mle_mean.pkl'.format(model_name,
                                                                                               noise_type,
                                                                                               sampling_distribution)),
                     'wb')
    pickle.dump(test_mle_mean, temp_file)
    temp_file.close()

    return test_data, test_mle_mean


def extract_predictions(clinical_map, mask, low_dimensional=False):
    """ Function to mask a 3D clinical volume, extracting values in a flattened, 1D format

        Inputs
        ------

        clinical_map : ndarray
            3 x N clinical map, where N is the number of values stored per voxel

        mask : ndarray
            3D binary mask, defining which voxels are to be extracted

        low_dimensional=False : bool, optional
            set to True is clinical map contains only one value per voxel

        Outputs
        -------
        clinical_map_flat_nonzero : ndarray
            array containing flattened, 1D masked values from clinical_map

        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    # flatten input 3D map
    if low_dimensional:
        clinical_map_flat = np.reshape(clinical_map,
                                       (clinical_map.shape[0] * clinical_map.shape[1] * clinical_map.shape[2], 1))
    else:
        clinical_map_flat = np.reshape(clinical_map, (
            clinical_map.shape[0] * clinical_map.shape[1] * clinical_map.shape[2], clinical_map.shape[3]))
    # flatten binary mask
    mask_flat = np.reshape(mask, (mask.shape[0] * mask.shape[1] * mask.shape[2]))
    # extract masked values
    clinical_map_flat_nonzero = clinical_map_flat[mask_flat != 0, :]

    return clinical_map_flat_nonzero


def import_training_data(SNR, script_dir, model_name, noise_type, sampling_distribution, dataset_size, SNR_alt):
    """ Function to load noise-free training data from disk to harmonise training across SNRs

        Inputs
        ------

        SNR : int
            signal-to-noise ratio

        script_dir : string
            base directory of main script

        model_name : string
            name of signal model being fit

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        dataset_size : int
            total number of signals generated for training and validation

        SNR_alt : int
            alternative SNR, used to harmonise initialisation if transfer_flag=True



        Outputs
        -------
        imported_data_train : method_classes.Dataset OR bool
            object containing training data and dataloaders OR False if no data is imported

        imported_data_val : method_classes.Dataset OR bool
            object containing training data and dataloaders OR False if no data is imported

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # harmonise noisefree training data across SNR
    if SNR == 15:

        # load training and validation datasets from disk
        temp_file = open(os.path.join(script_dir,
                                      'data/train/{}/{}/{}/train_data_{}_{}.pkl'.format(model_name,
                                                                                        noise_type,
                                                                                        sampling_distribution,
                                                                                        dataset_size,
                                                                                        SNR_alt)), 'rb')
        imported_data_train = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'data/train/{}/{}/{}/val_data_{}_{}.pkl'.format(model_name,
                                                                                      noise_type,
                                                                                      sampling_distribution,
                                                                                      dataset_size,
                                                                                      SNR_alt)), 'rb')
        imported_data_val = pickle.load(temp_file)
        temp_file.close()

    else:
        imported_data_train = False
        imported_data_val = False

    return imported_data_train, imported_data_val


def generate_training_data(train_dataset_flag, SNR, script_dir, model_name, noise_type, sampling_distribution, dataset_size,
                  SNR_alt, parameter_range, sampling_scheme):
    """ Function to generate training data

        Inputs
        ------

        train_dataset_flag : bool
            determine whether to generate training dataset; if False, load from disk

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        script_dir : string
            base directory of main script

        model_name : string
            name of signal model being fit

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        dataset_size : int
            total number of signals generated for training and validation

        SNR_alt : int
            alternative SNR, used to harmonise initialisation if transfer_flag=True

        parameter_range : ndarray
            defined in method_functions.get_parameter_scaling, provides boundaries for parameter_distribution

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme (independent
            variable values)


        Outputs
        -------
        training_data : method_classes.Dataset OR bool
            object containing training data and dataloaders

        validation_data : method_classes.Dataset OR bool
            object containing training data and dataloaders

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    if train_dataset_flag:
        print('Generating training data (SNR = {})...'.format(SNR))

        # import noise-free training data to harmonise across SNRs
        imported_data_train, imported_data_val = \
            import_training_data(SNR, script_dir, model_name, noise_type, sampling_distribution,
                                 dataset_size, SNR_alt)

        # generate training and validation datasets and save to disk
        train_data, validation_data = \
            create_dataset(script_dir=script_dir, model_name=model_name, parameter_distribution='uniform',
                           parameter_range=parameter_range, sampling_scheme=sampling_scheme,
                           sampling_distribution=sampling_distribution, noise_type=noise_type, SNR=SNR,
                           dataset_size=dataset_size, imported_data_train=imported_data_train,
                           imported_data_val=imported_data_val)
        print('...done!')
    else:
        print('Loading training data (SNR = {})...'.format(SNR))
        # load training and validation datasets from disk
        temp_file = open(os.path.join(script_dir,
                                      'data/train/{}/{}/{}/train_data_{}_{}.pkl'.format(model_name,
                                                                                        noise_type,
                                                                                        sampling_distribution,
                                                                                        dataset_size,
                                                                                        SNR)), 'rb')
        train_data = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'data/train/{}/{}/{}/val_data_{}_{}.pkl'.format(model_name,
                                                                                      noise_type,
                                                                                      sampling_distribution,
                                                                                      dataset_size,
                                                                                      SNR)), 'rb')
        validation_data = pickle.load(temp_file)
        temp_file.close()
        print('...done!')

    return train_data, validation_data


def networks(train_flag, SNR, transfer_flag, script_dir, network_arch, model_name, noise_type, sampling_distribution,
             dataset_size, SNR_alt, n_nets, training_data, validation_data, parameter_loss_scaling, criterion):
    """ Function to generate trained networks, either from scratch or loaded from disk

        Inputs
        ------

        train_flag : bool
            determines whether networks are trained (if True) or loaded from disk (if False)

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        transfer_flag : bool
            determines whether network initialisations are harmonised (if True)

        script_dir : string
            base directory of main script

        network_arch : string
            network architecture, as defined in method_functions.train_general_network

        model_name : string
            name of signal model being fit

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        dataset_size : int
            total number of signals generated for training and validation

        SNR_alt : int
            alternative SNR, used to harmonise initialisation if transfer_flag=True

        n_nets : int
            number of networks to train (with different random initialisations); if transfer_flag = True, only
            the best network is used for weight initialisation

        training_data : method_classes.Dataset
            object containing training data and dataloaders

        validation_data : method_classes.Dataset
            object containing validation data and dataloaders

        parameter_loss_scaling : ndarray
            relative weighting of each signal model parameter during supervised training; larger scaling
            results in lower weighting

        criterion : torch.nn loss criterion
            defines loss function used when training

        Outputs
        -------
        net_supervised_groundtruth : method_classes.trainedNet
            object containing trained network(s); supervised training, groundtruth labels

        net_supervised_mle : method_classes.trainedNet
            object containing trained network(s); supervised training, MLE labels

        net_supervised_mle_approx : method_classes.trainedNet
            object containing trained network(s); supervised training, MLE labels (incorrect noise model)

        net_selfsupervised : method_classes.trainedNet
            object containing trained network(s); selfsupervised training, sse loss

        net_supervised_hybrid : method_classes.trainedNet
            object containing trained network(s); supervised training, mixed weighting between MLE and GT labels

        net_supervised_mle_weighted_f : method_classes.trainedNet
            object containing trained network(s); supervised training, MLE labels, overweighted to f IVIM parameter

        net_supervised_mle_weighted_Dslow : method_classes.trainedNet
            object containing trained network(s); supervised training, MLE labels, overweighted to Dslow IVIM parameter

        net_supervised_mle_weighted_Dfast : method_classes.trainedNet
            object containing trained network(s); supervised training, MLE labels, overweighted to Dfast IVIM parameter


        Author: Sean C Epstein (https://seancepstein.github.io)
    """
    if train_flag:

        print('Training networks (SNR = {})...'.format(SNR))
        # harmonise starting network state across SNRs
        if transfer_flag and SNR == 15:
            transfer_flag_groundtruth = True
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

        else:
            state_dictionary = None
            transfer_flag_groundtruth = False

        net_supervised_groundtruth = train_general_network(
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
            transfer_learning=transfer_flag_groundtruth,
            state_dictionary=state_dictionary)

        # load state dictionary for transfer learning
        if SNR != 15:
            state_dictionary = net_supervised_groundtruth.best_network.state_dict()

        net_supervised_mle = train_general_network(
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

        net_supervised_mle_approx = train_general_network(
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

        net_selfsupervised = train_general_network(
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

        net_supervised_mle_weighted_f = train_general_network(
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

        net_supervised_mle_weighted_Dslow = train_general_network(
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

        net_supervised_mle_weighted_Dfast = train_general_network(
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

        net_supervised_hybrid = train_general_network(
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
        print('...done!')


    # otherwise, load networks from disk
    else:
        print('Loading trained networks (SNR = {})...'.format(SNR))

        temp_file = open(
            os.path.join(script_dir,
                         'models/{}/{}/{}/{}/supervised_groundtruth_{}_{}.pkl'.format(network_arch, model_name,
                                                                                      noise_type,
                                                                                      sampling_distribution,
                                                                                      dataset_size, SNR)), 'rb')
        net_supervised_groundtruth = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir,
                         'models/{}/{}/{}/{}/supervised_mle_{}_{}.pkl'.format(network_arch, model_name,
                                                                              noise_type,
                                                                              sampling_distribution,
                                                                              dataset_size, SNR)), 'rb')
        net_supervised_mle = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir,
                         'models/{}/{}/{}/{}/supervised_mle_approx_{}_{}.pkl'.format(network_arch, model_name,
                                                                                     noise_type,
                                                                                     sampling_distribution,
                                                                                     dataset_size, SNR)), 'rb')
        net_supervised_mle_approx = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir,
                         'models/{}/{}/{}/{}/selfsupervised_{}_{}.pkl'.format(network_arch, model_name,
                                                                              noise_type, sampling_distribution,
                                                                              dataset_size, SNR)), 'rb')
        net_selfsupervised = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir,
                         'models/{}/{}/{}/{}/selfsupervised_rician_{}_{}.pkl'.format(network_arch, model_name,
                                                                                     noise_type, sampling_distribution,
                                                                                     dataset_size, SNR)), 'rb')

        net_supervised_hybrid = pickle.load(temp_file)
        temp_file.close()

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
        print('...done!')

    return net_supervised_groundtruth, net_supervised_mle, net_supervised_mle_approx, net_selfsupervised, \
        net_supervised_hybrid, net_supervised_mle_weighted_f, net_supervised_mle_weighted_Dslow, \
        net_supervised_mle_weighted_Dfast


def test_data_uniform(test_dataset_flag, SNR, model_name, script_dir, noise_type, sampling_distribution, dataset_size,
                      training_data):
    """ Function to either generate testing data or load previously generated data from disk

        Inputs
        ------
        test_dataset_flag : bool
            if True, generate training data; if False, load from disk

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        model_name : string
            name of signal model being fit


        script_dir : string
            base directory of main script

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        dataset_size : int
            total number of signals generated for training and validation

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        training_data : method_classes.Dataset
            object containing training data and dataloaders

        Outputs
        -------
        test_data : method_classes.testDataset
            object containing test data

        test_mle_groundtruth : method_classes.testResults
            fitting performance of conventional fitting, initialised with groundtruth parameter values

        test_mle_mean : method_classes.testResults
            fitting performance of conventional fitting, initialised with tissue-mean parameter values


        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    if test_dataset_flag:
        print('Generating testing data (SNR = {})...'.format(SNR))
        # generate test dataset and compute MLE baseline
        test_data, test_mle_groundtruth, test_mle_mean = \
            create_test_data(training_data=training_data, n_repeats=500, n_sampling=[1, 10, 10, 10],
                             extent_scaling=0.0, model_name=model_name, script_dir=script_dir, noise_type=noise_type,
                             sampling_distribution=sampling_distribution, dataset_size=dataset_size, SNR=SNR)
        print('...done!')
    else:
        print('Loading testing data (SNR = {})...'.format(SNR))
        # load test dataset + MLE baseline from disk
        temp_file = open(os.path.join(script_dir,
                                      'data/test/{}/{}/{}/test_data_{}_{}.pkl'.format(model_name,
                                                                                      noise_type,
                                                                                      sampling_distribution,
                                                                                      dataset_size,
                                                                                      SNR)), 'rb')
        test_data = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'data/test/{}/{}/{}/test_MLE_groundtruth_{}_{}.pkl'.format(model_name,
                                                                                                 noise_type,
                                                                                                 sampling_distribution,
                                                                                                 dataset_size,
                                                                                                 SNR)), 'rb')
        test_mle_groundtruth = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'data/test/{}/{}/{}/test_MLE_mean_{}_{}.pkl'.format(model_name,
                                                                                          noise_type,
                                                                                          sampling_distribution,
                                                                                          dataset_size,
                                                                                          SNR)), 'rb')
        test_mle_mean = pickle.load(temp_file)
        temp_file.close()
        print('...done!')

    return test_data, test_mle_groundtruth, test_mle_mean


def evaluate_methods(test_flag, SNR, model_name, test_data, sampling_distribution, script_dir, noise_type, dataset_size,
                     network_arch, test_mle_groundtruth, test_mle_mean, net_supervised_groundtruth, net_supervised_mle,
                     net_supervised_mle_approx, net_selfsupervised, net_supervised_hybrid,
                     net_supervised_mle_weighted_f, net_supervised_mle_weighted_Dslow,
                     net_supervised_mle_weighted_Dfast):
    """ Function to either evaluate parameter estimations or load previous results from disk

        Inputs
        ------
        test_flag : bool
            if True, evaluate merhods; if False, load from disk

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        model_name : string
            name of signal model being fit

        test_data : method_classes.testDataset
            object containing test data

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        script_dir : string
            base directory of main script

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        dataset_size : int
            total number of signals generated for training and validation

        network_arch : string
            network architecture, as defined in method_functions.train_general_network

        test_mle_groundtruth : method_classes.testResults
            object containing conventional MLE fits (seeded with groundtruth values) of the test data

        test_mle_mean : method_classes.testResults
            object containing conventional MLE fits (seeded with mean values) of the test data

        net_supervised_groundtruth : method_classes.trainedNet
            object containing network(s) trained with in a supervised manner with groundtruth labels

        net_supervised_mle : method_classes.trainedNet
            object containing network(s) trained with in a supervised manner with MLE labels using the correct
            noise model

        net_supervised_mle_approx : method_classes.trainedNet
            object containing network(s) trained with in a supervised manner with MLE labels using the incorrect
            noise model

        net_selfsupervised : method_classes.trainedNet
            object containing network(s) trained with in a selfsupervised manner

        net_supervised_hybrid : method_classes.trainedNet
            object containing network(s) trained with in a supervised manner using a weighted sum of MLE and
            groundtruth labels

        net_supervised_mle_weighted_f : method_classes.trainedNet
            object containing network(s) trained to predict f in a supervised manner with MLE labels

        net_supervised_mle_weighted_Dslow : method_classes.trainedNet
            object containing network(s) trained to predict Dslow in a supervised manner with MLE labels

        net_supervised_mle_weighted_Dfast : method_classes.trainedNet
            object containing network(s) trained to predict Dfast in a supervised manner with MLE labels

        Outputs
        -------
        test_mle_groundtruth : method_classes.testResults
            fitting performance of conventional fitting, initialised with groundtruth parameter values

        test_mle_mean : method_classes.testResults
            fitting performance of conventional fitting, initialised with tissue-mean parameter values

        test_supervised_groundtruth : method_classes.testResults
            fitting performance of network fitting, supervised training, groundtruth labels

        test_supervised_mle : method_classes.testResults
            fitting performance of network fitting, supervised training, MLE labels

        test_supervised_mle_approx : method_classes.testResults
            fitting performance of network fitting, supervised training, MLE labels (incorrect noise model)

        test_selfsupervised : method_classes.testResults
            fitting performance of network fitting, selfsupervised training

        test_hybrid : method_classes.testResults
            fitting performance of network fitting, supervised training, mixed weighting between MLE and groundtruth
            labels

        test_mle_weighted_f : method_classes.testResults
            fitting performance of network fitting, supervised training, MLE labels, overweighted to f IVIM
            parameter

        test_mle_weighted_Dslow : method_classes.testResults
            fitting performance of network fitting, supervised training, MLE labels, overweighted to Dslow IVIM
            parameter

        test_mle_weighted_Dfast : method_classes.testResults
            fitting performance of network fitting, supervised training, MLE labels, overweighted to Dfast IVIM
            parameter

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    if test_flag:
        print('Testing networks (SNR = {})...'.format(SNR))
        # evaluate quality metrics for conventional fitting
        test_mle_groundtruth = test_performance_wrapper(
            network_type='mle_groundtruth',
            model_name=model_name,
            network=None,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch,
            mle_flag=True,
            mle_data=test_mle_groundtruth)

        test_mle_mean = test_performance_wrapper(
            network_type='mle_mean',
            model_name=model_name,
            network=None,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch,
            mle_flag=True,
            mle_data=test_mle_mean)

        # compute test performance of networks
        test_supervised_groundtruth = test_performance_wrapper(
            network_type='supervised_groundtruth',
            model_name=model_name,
            network=net_supervised_groundtruth,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch)

        test_supervised_mle = test_performance_wrapper(
            network_type='supervised_mle',
            model_name=model_name,
            network=net_supervised_mle,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch)

        test_supervised_mle_approx = test_performance_wrapper(
            network_type='supervised_mle_approx',
            model_name=model_name,
            network=net_supervised_mle_approx,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch)

        test_selfsupervised = test_performance_wrapper(
            network_type='selfsupervised',
            model_name=model_name,
            network=net_selfsupervised,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch)

        test_hybrid = test_performance_wrapper(
            network_type='hybrid',
            model_name=model_name,
            network=net_supervised_hybrid,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch)

        test_mle_weighted_f = test_performance_wrapper(
            network_type='mle_weighted_f',
            model_name=model_name,
            network=net_supervised_mle_weighted_f,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch)
        test_mle_weighted_Dslow = test_performance_wrapper(
            network_type='mle_weighted_Dslow',
            model_name=model_name,
            network=net_supervised_mle_weighted_Dslow,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch)
        test_mle_weighted_Dfast = test_performance_wrapper(
            network_type='mle_weighted_Dfast',
            model_name=model_name,
            network=net_supervised_mle_weighted_Dfast,
            test_data=test_data,
            sampling_distribution=sampling_distribution,
            script_dir=script_dir,
            noise_type=noise_type,
            dataset_size=dataset_size,
            SNR=SNR,
            network_arch=network_arch)

        print('...done!')
    else:
        print('Loading network test results (SNR = {})...'.format(SNR))
        # load test performance from disk

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_mle_groundtruth_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_mle_groundtruth = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_mle_mean_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_mle_mean = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_supervised_groundtruth_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_supervised_groundtruth = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_supervised_mle_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_supervised_mle = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_supervised_mle_approx_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_supervised_mle_approx = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_selfsupervised_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_selfsupervised = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_hybrid_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_hybrid = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_mle_weighted_f_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_mle_weighted_f = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_mle_weighted_Dslow_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_mle_weighted_Dslow = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir,
                                      'results/{}/{}/{}/{}/test_mle_weighted_Dfast_{}_{}.pkl'.format(
                                          network_arch, model_name, noise_type, sampling_distribution,
                                          dataset_size, SNR)), 'rb')
        test_mle_weighted_Dfast = pickle.load(temp_file)
        temp_file.close()

    print('...done!')

    return test_mle_groundtruth, test_mle_mean, test_supervised_groundtruth, test_supervised_mle, \
        test_supervised_mle_approx, test_selfsupervised, test_hybrid, test_mle_weighted_f, test_mle_weighted_Dslow, \
        test_mle_weighted_Dfast


def test_data_clinical(clinical_data_flag, script_dir, model_name, noise_type, sampling_scheme, sampling_distribution,
                       SNR, net_supervised_groundtruth, net_supervised_mle, net_supervised_mle_approx,
                       net_selfsupervised):
    """ Function to analyse clinical data and evaluate fitting performance on it

        Inputs
        ------
        clinical_data_flag : bool
            if True, process clinical data; if False, load outputs from disk

        script_dir : string
            base directory of main script

        model_name : string
            name of signal model being fit

        noise_type: string
            defined in method_functions.add_noise, name of noise type

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme (independent
            variable values)

        sampling_distribution : string
            defined in method_functions.get_sampling_scheme, name of sampling distribution

        SNR : int
            defined in method_functions.add_noise, signal to noise ratio

        net_supervised_groundtruth : method_classes.trainedNet
            object containing network(s) trained with in a supervised manner with groundtruth labels

        net_supervised_mle : method_classes.trainedNet
            object containing network(s) trained with in a supervised manner with MLE labels using the correct
            noise model

        net_supervised_mle_approx : method_classes.trainedNet
            object containing network(s) trained with in a supervised manner with MLE labels using the incorrect
            noise model

        net_selfsupervised : method_classes.trainedNet
            object containing network(s) trained with in a selfsupervised manner


        Outputs
        -------
        test_clinical_real_mle_mean : method_classes.testResults
            conventional MLE (initialised with tissue-mean) estimates of the clinical data

        test_clinical_real_supervised_groundtruth : method_classes.testResults
            supervised training (groundtruth labels) estimates of the clinical data

        test_clinical_real_supervised_mle : method_classes.testResults
            supervised training (MLE labels) estimates of the clinical data

        test_clinical_real_supervised_mle_approx : method_classes.testResults
            supervised training (MLE labels, incorrect noise model) estimates of the clinical data

        test_clinical_real_selfsupervised : method_classes.testResults
            selfsupervised training estimates of the clinical data

        test_clinical_synth_mle_mean : method_classes.testResults
            conventional MLE (initialised with tissue-mean) estimates of the synthetic clinical data

        test_clinical_synth_supervised_groundtruth : method_classes.testResults
            supervised training (groundtruth labels) estimates of the synthetic clinical data

        test_clinical_synth_supervised_mle : method_classes.testResults
            supervised training (MLE labels) estimates of the synthetic clinical data

        test_clinical_synth_supervised_mle_approx : method_classes.testResults
            supervised training (MLE labels, incorrect noise model) estimates of the synthetic clinical data

        test_clinical_synth_selfsupervised : method_classes.testResults
            selfsupervised training estimates of the synthetic clinical data

        masked_groundtruth : ndarray
            flattened clinical groundtruth labels, masked to remove bowels

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    print('Importing clinical data...')
    # import data from volunteer
    img = nibabel.load(os.path.join(script_dir, 'clinical_data/epstein_all_data.nii')).get_fdata()
    # binary mask, removing background voxels
    mask = nibabel.load(os.path.join(script_dir, 'clinical_data/epstein_background_mask.nii')).get_fdata()
    # binary mask, removing background voxels plus bowel
    mask_no_bowel = nibabel.load(os.path.join(script_dir, 'clinical_data/epstein_bowel_mask.nii')).get_fdata()
    # number of non-background voxels
    n_voxels = np.count_nonzero(mask)
    # number of b-values
    n_bvals = 160

    # reshape image into 1D array
    img_1D = np.zeros((n_voxels, n_bvals))
    # store b=0 standard deviation at each voxel value
    img_std = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    # store indices for 1D <-> 3D mapping
    idx_3D = np.zeros((img.shape[0] + 1, img.shape[1] + 1, img.shape[2] + 1))
    idx_1D = np.zeros((3, n_voxels))
    # convert imported data into 1D array
    count = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for z in range(img.shape[2]):
                if mask[x, y, z] > 0:
                    img_1D[count, :] = img[x, y, z, :]
                    idx_1D[:, count] = [x + 1, y + 1, z + 1]
                    idx_3D[x + 1, y + 1, z + 1] = count
                    img_std[x, y, z] = np.std(img_1D[count, 0:16])
                    count = count + 1
    print('...done!')

    # calculate standard deviation across b=0 images, i.e. estimate SNR for each voxel
    img_1D_std = np.std(img_1D[:, 0:16], axis=1)

    # subsample dataset to obtain realistic acquisitions
    print('Subsampling dataset to obtain clinical acquisition protocol...')
    epstein_subsampled = np.empty((img_1D.shape[0], 10, 16))
    epstein_subsampled_norm = np.empty((img_1D.shape[0], 10, 16))

    for subsample in range(16):
        epstein_subsampled[:, :, subsample] = img_1D[:, subsample::16]
        epstein_subsampled_norm[:, :, subsample] = np.divide(epstein_subsampled[:, :, subsample],
                                                             np.matlib.repmat(epstein_subsampled[:, 0, subsample], 10,
                                                                              1).T)
    print('...done!')

    # calculate 'groundtruth' by computing MLE on complete dataset
    if clinical_data_flag:
        print('Calculating ''groundtruth'' parameter values from supersampled dataset...')
        fitting_seed = np.zeros((n_voxels, 4))
        fitting_seed[:, 0] = np.mean(img_1D[:, 0:16], axis=1)
        fitting_seed[:, 1] = 0.3
        fitting_seed[:, 2] = 1.5
        fitting_seed[:, 3] = 50

        clinical_bestfit_mle = traditional_fitting(model_name='ivim', signal=img_1D,
                                                   sampling_scheme=get_sampling_scheme('ivim_160'),
                                                   labels_groundtruth=fitting_seed, SNR=None, noise_type='rician',
                                                   seed_mean=False, calculate_sigma=True)
        parameter_map_mle = np.zeros((img.shape[0], img.shape[1], img.shape[2], 4))
        likelihood_map_mle = np.zeros((img.shape[0], img.shape[1], img.shape[2], 1))
        rmse_signal_residuals_mle = np.zeros((img.shape[0], img.shape[1], img.shape[2], 1))
        count = 0
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                for z in range(img.shape[2]):
                    if mask[x, y, z] > 0:
                        parameter_map_mle[x, y, z, :] = clinical_bestfit_mle[count, :]
                        likelihood_map_mle[x, y, z, :] = rician_log_likelihood(generate_IVIM(
                            clinical_bestfit_mle[count, :], get_sampling_scheme('ivim_160')), img_1D[count, :],
                            sigma=img_1D_std[count])
                        predicted_signal = generate_IVIM(parameter_map_mle[x, y, z, :], get_sampling_scheme('ivim_160'))
                        rmse_signal_residuals_mle[x, y, z, :] = np.sqrt(
                            np.mean(np.square((img[x, y, z, :] - predicted_signal))))
                        count = count + 1

        temp_file = open(os.path.join(script_dir, 'clinical_data/epstein_params_groundtruth_mle.pkl'),
                         'wb')

        pickle.dump(parameter_map_mle, temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir, 'clinical_data/epstein_likelihood_groundtruth_mle.pkl'), 'wb')

        pickle.dump(likelihood_map_mle, temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir, 'clinical_data/epstein_rmse_residuals_groundtruth_mle.pkl'), 'wb')

        pickle.dump(rmse_signal_residuals_mle, temp_file)
        temp_file.close()
    else:
        print('Loading ''groundtruth'' parameter values calculated from supersampled dataset...')
        temp_file = open(os.path.join(script_dir, 'clinical_data/epstein_params_groundtruth_mle.pkl'),
                         'rb')

        parameter_map_mle = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir, 'clinical_data/epstein_likelihood_groundtruth_mle.pkl'), 'rb')

        likelihood_map_mle = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir, 'clinical_data/epstein_rmse_residuals_groundtruth_mle.pkl'), 'rb')

        rmse_signal_residuals_mle = pickle.load(temp_file)
        temp_file.close()
        print('...done!')

    # calculate/load MLE of subsampled datasets
    if clinical_data_flag:
        print('Calculating MLE estimates of subsampled dataset...')
        fitting_seed = np.zeros((n_voxels, 4))
        fitting_seed[:, 1] = 0.3
        fitting_seed[:, 2] = 1.5
        fitting_seed[:, 3] = 50

        # construct parameter and likelihood 3D spatial maps
        epstein_subsampled_mle_mean = np.zeros((epstein_subsampled.shape[0], 4, 16))
        subsampled_parameter_map_mle_mean = np.zeros((img.shape[0], img.shape[1], img.shape[2], 4, 16))
        subsampled_likelihood_map_mle_mean = np.zeros((img.shape[0], img.shape[1], img.shape[2], 16))
        for subsample in range(16):
            fitting_seed[:, 0] = epstein_subsampled[:, 0, subsample]
            # calculate MLE
            epstein_subsampled_mle_mean[:, :, subsample] = traditional_fitting(model_name='ivim', signal=np.squeeze(
                epstein_subsampled[:, :, subsample]), sampling_scheme=get_sampling_scheme('ivim_10'),
                                                                               labels_groundtruth=fitting_seed,
                                                                               SNR=None, noise_type='rician',
                                                                               seed_mean=False,
                                                                               calculate_sigma=False,
                                                                               sigma_array=img_1D_std)
            # save in 3D map
            count = 0
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    for z in range(img.shape[2]):
                        if mask[x, y, z] > 0:
                            subsampled_parameter_map_mle_mean[x, y, z, :, subsample] = epstein_subsampled_mle_mean[
                                                                                       count, :, subsample]
                            subsampled_likelihood_map_mle_mean[x, y, z, subsample] = rician_log_likelihood(
                                generate_IVIM(epstein_subsampled_mle_mean[count, :, subsample],
                                              get_sampling_scheme('ivim_10')),
                                np.squeeze(epstein_subsampled[count, :, subsample]), sigma=img_1D_std[count])
                            count = count + 1

        temp_file = open(os.path.join(script_dir, 'clinical_data/subsampled_parameter_map_mle_mean.pkl'),
                         'wb')
        pickle.dump(subsampled_parameter_map_mle_mean, temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir, 'clinical_data/subsampled_likelihood_map_mle_mean.pkl'), 'wb')

        pickle.dump(subsampled_likelihood_map_mle_mean, temp_file)
        temp_file.close()

        print('...done!')
    else:
        print('Load MLE estimates of subsampled dataset...')
        # load from disk
        temp_file = open(os.path.join(script_dir, 'clinical_data/subsampled_parameter_map_mle_mean.pkl'),
                         'rb')
        subsampled_parameter_map_mle_mean = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(
            os.path.join(script_dir, 'clinical_data/subsampled_likelihood_map_mle_mean.pkl'), 'rb')
        subsampled_likelihood_map_mle_mean = pickle.load(temp_file)
        temp_file.close()
        print('...done!')

    print('Evaluating networks on clinical data...'.format(SNR))
    # reshape subsampled voxels into 3D spatial maps
    epstein_subsampled_3D = np.zeros((224, 224, 5, 16, 10))
    count = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for z in range(img.shape[2]):
                if mask[x, y, z] > 0:
                    epstein_subsampled_3D[x, y, z, :, :] = epstein_subsampled_norm[count, :, :].T
                    count = count + 1

    # mask maps to remove bowel and background voxels
    epstein_subsampled_3D_masked = np.reshape(extract_predictions(np.reshape(epstein_subsampled_3D,
                                                                             (224, 224, 5, 160)), mask_no_bowel),
                                              (np.count_nonzero(mask_no_bowel), 16, 10))
    # reshape masked maps into flat arrays
    epstein_subsampled_1D_masked = np.reshape(epstein_subsampled_3D_masked,
                                              (epstein_subsampled_3D_masked.shape[0] * 16, 10))

    masked_groundtruth = extract_predictions(parameter_map_mle, mask_no_bowel)

    test_clinical_real_supervised_groundtruth = test_network(
        model_name=model_name, test_data=epstein_subsampled_1D_masked,
        test_label=masked_groundtruth,
        n_sampling=np.count_nonzero(mask_no_bowel),
        n_repeats=16, network=net_supervised_groundtruth.best_network, clinical_flag=True)

    test_clinical_real_supervised_mle = test_network(
        model_name=model_name, test_data=epstein_subsampled_1D_masked,
        test_label=masked_groundtruth,
        n_sampling=np.count_nonzero(mask_no_bowel),
        n_repeats=16, network=net_supervised_mle.best_network, clinical_flag=True)

    test_clinical_real_supervised_mle_approx = test_network(
        model_name=model_name, test_data=epstein_subsampled_1D_masked,
        test_label=masked_groundtruth,
        n_sampling=np.count_nonzero(mask_no_bowel),
        n_repeats=16, network=net_supervised_mle_approx.best_network, clinical_flag=True)

    test_clinical_real_selfsupervised = test_network(
        model_name=model_name, test_data=epstein_subsampled_1D_masked,
        test_label=masked_groundtruth,
        n_sampling=np.count_nonzero(mask_no_bowel),
        n_repeats=16, network=net_selfsupervised.best_network, clinical_flag=True)

    test_clinical_real_mle_mean = test_network(
        model_name=model_name, test_data=epstein_subsampled_1D_masked,
        test_label=masked_groundtruth,
        n_sampling=np.count_nonzero(mask_no_bowel),
        n_repeats=16, network=net_selfsupervised.best_network, clinical_flag=True, mle_flag=True,
        mle_data=method_classes.testResults(
            network_object=None, test_data=None, param_predictions=np.swapaxes(np.reshape(
                extract_predictions(np.reshape(subsampled_parameter_map_mle_mean, (224, 224, 5, 4 * 16)),
                                    mask_no_bowel), (np.count_nonzero(mask_no_bowel), 4, 16)), 1, 2),
            param_predictions_mean=None, param_predictions_rmse=None, param_predictions_std=None))

    print('...done!')

    # generate/load synthetic data matching clinical groundtruth distribution
    if clinical_data_flag:
        print('Generating synthetic test data based on clinical parameter distribution...')
        test_clinical_synth_dataset, test_clinical_synth_mle_mean = \
            create_test_data_clinical(label=masked_groundtruth, n_repeats=16,
                                      model_name=model_name, script_dir=script_dir, noise_type=noise_type,
                                      sampling_scheme=sampling_scheme,
                                      sampling_distribution=sampling_distribution,
                                      sigma=extract_predictions(img_std, mask_no_bowel, low_dimensional=True))
    else:
        print('Loading synthetic test data based on clinical parameter distribution...')

        temp_file = open(os.path.join(script_dir, 'data/test/{}/{}/{}/test_clinical_synth_dataset.pkl'.format(
            model_name, noise_type, sampling_distribution)), 'rb')
        test_clinical_synth_dataset = pickle.load(temp_file)
        temp_file.close()

        temp_file = open(os.path.join(script_dir, 'data/test/{}/{}/{}/test_clinical_synth_mle_mean.pkl'.format(
            model_name, noise_type, sampling_distribution)), 'rb')
        test_clinical_synth_mle_mean = pickle.load(temp_file)
        temp_file.close()

    print('...done!')

    print('Evaluating networks on clinically-derived synthetic test data (SNR = {})...'.format(SNR))
    # normalise test data by b=0 values
    test_data_noisy_flat_norm = np.divide(test_clinical_synth_dataset.test_data_noisy_flat,
                                          np.matlib.repmat(test_clinical_synth_dataset.test_data_noisy_flat[:, 0], 10,
                                                           1).T)

    test_clinical_synth_supervised_groundtruth = test_network(
        model_name=model_name, test_data=test_data_noisy_flat_norm,
        test_label=test_clinical_synth_dataset.test_label,
        n_sampling=test_clinical_synth_dataset.test_data.shape[0],
        n_repeats=16, network=net_supervised_groundtruth.best_network, clinical_flag=True)
    test_clinical_synth_supervised_mle = test_network(
        model_name=model_name, test_data=test_data_noisy_flat_norm,
        test_label=test_clinical_synth_dataset.test_label,
        n_sampling=test_clinical_synth_dataset.test_data.shape[0],
        n_repeats=16, network=net_supervised_mle.best_network, clinical_flag=True)
    test_clinical_synth_supervised_mle_approx = test_network(
        model_name=model_name, test_data=test_data_noisy_flat_norm,
        test_label=test_clinical_synth_dataset.test_label,
        n_sampling=test_clinical_synth_dataset.test_data.shape[0],
        n_repeats=16, network=net_supervised_mle_approx.best_network, clinical_flag=True)
    test_clinical_synth_selfsupervised = test_network(
        model_name=model_name, test_data=test_data_noisy_flat_norm,
        test_label=test_clinical_synth_dataset.test_label,
        n_sampling=test_clinical_synth_dataset.test_data.shape[0],
        n_repeats=16, network=net_selfsupervised.best_network, clinical_flag=True)
    print('...done!')

    return test_clinical_real_mle_mean, test_clinical_real_supervised_groundtruth, \
        test_clinical_real_supervised_mle, test_clinical_real_supervised_mle_approx, \
        test_clinical_real_selfsupervised, \
        test_clinical_synth_mle_mean, test_clinical_synth_supervised_groundtruth, \
        test_clinical_synth_supervised_mle, test_clinical_synth_supervised_mle_approx, \
        test_clinical_synth_selfsupervised, masked_groundtruth


def traditional_fitting_create_data(model_name, signal, sampling_scheme, labels_groundtruth, SNR, noise_type,
                                    seed_mean=False, calculate_sigma=False, sigma_array=None, segmented_flag=False):
    """ Function to compute conventional MLE when creating training/validation datasets

            Inputs
            ------

            model_name : string
                name of signal model being fit

            signal : ndarray
                signal to fit model to

            sampling_scheme : ndarray
                defined in method_functions.get_sampling_scheme, provides signal sampling scheme (independent
                variable values)

            labels_groundtruth : ndarray
                groundtruth generative parameters

            SNR : int
                defined in method_functions.add_noise, signal to noise ratio

            noise_type: string
                defined in method_functions.add_noise, name of noise type

            seed_mean : optional, bool
                if True, seeds fitting with mean parameter values; if False, seeds with groundtruth values

            calculate_sigma : optional, bool
                if True, estimate signal standard deviation from b=0 independently for each model fit

            sigma_array : optional, ndarray
                standard deviation corresponding to each signal

            segmented_flag : optional, bool
                if True, use segmented fitting as described in doi:10.1002/jmri.24799

            Outputs
            -------
            labels_bestfit : ndarray
                best fit MLE parameters

            Author: Sean C Epstein (https://seancepstein.github.io)
        """
    # generate empty object to store bestfit parameters
    labels_bestfit = np.zeros_like(labels_groundtruth)
    labels_bestfit_all = np.zeros((labels_groundtruth.shape[0], 4, 3))
    loss_bestfit_all = np.full((labels_groundtruth.shape[0], 3), np.inf)
    # number of signals being fit
    # n_signals = 2000
    n_signals = signal.shape[0]
    if model_name == 'ivim':

        if seed_mean:
            # determine fitting seed, set to mean parameter value
            seed = np.ones_like(labels_groundtruth) * labels_groundtruth.mean(axis=0)
        else:
            seed = labels_groundtruth

        # set upper and lower bounds for fitting
        # bounds = Bounds(lb=np.array([0, 0.05, 0.05, 2.0]), ub=np.array([10000, 0.5, 4, 300]))
        bounds = Bounds(lb=np.array([0, 0.01, 0.01, 2.0]), ub=np.array([10000, 0.5, 4, 300]))
        # perform fitting
        start = time.time()
        print_timer = 0
        for training_data in range(n_signals):

            if calculate_sigma:
                SNR = 1 / np.std(signal[training_data, :][sampling_scheme == 0])
            if sigma_array is not None:
                SNR = 1 / sigma_array[training_data]

            if segmented_flag:
                threshold = 0.05
                threshold_idx = sampling_scheme > threshold

                # take natural log of noisy data to perform weighted linear regression
                log_signal = np.log(signal[training_data, threshold_idx])
                # compute design matrix for initial (non-weighted) linear regression
                designMatrix = np.concatenate(
                    (sampling_scheme[threshold_idx, np.newaxis],
                     np.ones_like(sampling_scheme[threshold_idx, np.newaxis])),
                    axis=1)
                # compute first-guess non-weighted coefficients
                negADC, lnS0 = np.linalg.lstsq(designMatrix, np.transpose(log_signal), rcond=None)[0]
                # perform weighted least squares, using first-guess (MLE signal)^2 as weights

                # compute first-pass MLE signal
                initial_guess = generate_ADC(label=[np.exp(lnS0), -negADC],
                                            sampling_scheme=sampling_scheme[threshold_idx])
                # compute weights as MLE signal **2
                sqrt_weights = np.sqrt(np.diag(initial_guess ** 2))
                # multiply design matrix and log-signal by weights
                designMatrix_weighted = np.dot(sqrt_weights, designMatrix)
                log_signal_weighted = np.dot(log_signal, sqrt_weights)
                # second pass weighted linear least squares
                negADC_weighted, lnS0_weighted = \
                    np.linalg.lstsq(designMatrix_weighted, log_signal_weighted, rcond=None)[0]

                # fix Dslow to ADC value
                labels_bestfit[training_data, 2] = -negADC_weighted

                # compute MLE over 3 other model parameters
                bounds_segmented = Bounds(lb=np.array([0, 0.01, 2.0]), ub=np.array([10000, 0.5, 300]))
                optimize_result = minimize(fun=cost_gradient_descent, x0=seed[training_data, [0, 1, 3]],
                                           bounds=bounds_segmented, args=[signal[training_data, :], SNR,
                                                                          'ivim_segmented', noise_type, sampling_scheme,
                                                                          labels_bestfit[training_data, 2]])

                labels_bestfit[training_data, 0] = optimize_result.x[0]
                labels_bestfit[training_data, 1] = optimize_result.x[1]
                labels_bestfit[training_data, 3] = optimize_result.x[2]
                loss_bestfit_all[training_data, 0] = optimize_result.fun

            else:
                # compute + save model parameters
                optimize_result = minimize(fun=cost_gradient_descent, x0=seed[training_data, :],
                                           bounds=bounds, args=[signal[training_data, :], SNR,
                                                                model_name, noise_type, sampling_scheme])

                labels_bestfit[training_data, :] = optimize_result.x
                labels_bestfit_all[training_data, :, 0] = optimize_result.x
                loss_bestfit_all[training_data, 0] = optimize_result.fun

                elapsed_seconds = time.time() - start

                if int(elapsed_seconds) > print_timer:
                    sys.stdout.write('\r')
                    sys.stdout.write('MLE progress: {:.2f}%, elapsed time: {}'.format(training_data / n_signals * 100,
                                                                                      str(datetime.timedelta(
                                                                                          seconds=elapsed_seconds)).split(
                                                                                          ".")[0]))
                    sys.stdout.flush()
                    print_timer = print_timer + 1
                elif training_data == n_signals - 1:
                    sys.stdout.write('\r')
                    sys.stdout.write('...MLE done!'.format(training_data / n_signals * 100))
                    sys.stdout.flush()
        sys.stdout.write('\r')

    else:
        sys.exit("Implement other signal models here")
    return labels_bestfit, loss_bestfit_all, labels_bestfit_all


def generate_ADC(label, sampling_scheme):
    """ Function to generate ADC signal

        Inputs
        ------
        label : ndarray
            groundtruth generative parameters

        sampling_scheme : ndarray
            defined in method_functions.get_sampling_scheme, provides signal sampling scheme
            (independent variable values)

        Outputs
        -------
        signal : ndarray
            noisefree signal

        Author: Sean C Epstein (https://seancepstein.github.io)
    """

    # read model parameters from numpy label object
    s0 = label[0]
    ADC = label[1]
    # generate numpy signal from model parameters
    signal = s0 * np.exp(np.dot(-sampling_scheme, ADC))
    return signal