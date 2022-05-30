import torch, pickle, random, itertools, os, pathlib, sys
import method_classes
import numpy as np
import numpy.matlib as matlib
import torch.optim as optim
import torch.utils.data as utils
import numpy.ma as ma
from multiprocessing import Pool
from scipy.optimize import minimize, Bounds
from scipy import special


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
    else:
        sys.exit("Implement other sampling schemes in method_functions.get_sampling_scheme")
    return sampling_scheme


def get_parameter_scaling(model_name):
    """ Function to generate generative parameter range + distribution associated with a given signal model

            Inputs
            ------

            model_name : string
                name of signal model being fit

            Outputs
            -------
            parameter_range : ndarray
                provides boundaries for parameter_distribution

            parameter_loss_scaling : ndarray
                relative weighting of each signal model parameter during supervised training; larger scaling
                results in lower weighting

            Author: Sean C Epstein (https://seancepstein.github.io)
        """

    if model_name == 'ivim':
        parameter_range = np.array([[0.8, 1.2],  # S0
                                    [0, 0.8],  # f
                                    [0, 4.0],  # Dslow
                                    [0, 80]])  # Dfast
        parameter_loss_scaling = [1, 0.25, 1.25, 30]
        # parameter_range = np.array([[0.8, 1.2],  # S0
        #                             [0.1, 0.4],  # f
        #                             [0.5, 2.0],  # Dslow
        #                             [10, 50]])  # Dfast
        # parameter_loss_scaling = [1, 0.25, 1.25, 30]
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
    # add noise to training data
    train_signal_noisy = add_noise(noise_type=noise_type,
                                   SNR=SNR,
                                   signal=train_signal)

    # add noise to validation data
    val_signal_noisy = add_noise(noise_type=noise_type,
                                 SNR=SNR,
                                 signal=val_signal)

    # calculate MLE estimates of training data
    train_label_bestfit = traditional_fitting(model_name=model_name,
                                              signal=train_signal_noisy,
                                              sampling_scheme=sampling_scheme,
                                              labels_groundtruth=train_label_groundtruth,
                                              SNR=SNR,
                                              noise_type=noise_type,
                                              seed_mean=False)

    # calculate MLE estimates of validation data
    val_label_bestfit = traditional_fitting(model_name=model_name,
                                            signal=val_signal_noisy,
                                            sampling_scheme=sampling_scheme,
                                            labels_groundtruth=val_label_groundtruth,
                                            SNR=SNR,
                                            noise_type=noise_type,
                                            seed_mean=False)

    if model_name == 'ivim' and noise_type == 'rician':
        # calculate MLE using gaussian noise model, despite rician noise
        train_label_bestfit_approx = traditional_fitting(model_name=model_name,
                                                         signal=train_signal_noisy,
                                                         sampling_scheme=sampling_scheme,
                                                         labels_groundtruth=train_label_groundtruth,
                                                         SNR=SNR,
                                                         noise_type='gaussian',
                                                         seed_mean=False)

        # calculate MLE using gaussian noise model, despite rician noise
        val_label_bestfit_approx = traditional_fitting(model_name=model_name,
                                                       signal=val_signal_noisy,
                                                       sampling_scheme=sampling_scheme,
                                                       labels_groundtruth=val_label_groundtruth,
                                                       SNR=SNR,
                                                       noise_type='gaussian',
                                                       seed_mean=False)
    else:
        train_label_bestfit_approx = None
        val_label_bestfit_approx = None

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
            torch.Tensor(np.stack([train_label_bestfit, train_label_groundtruth], axis=2))),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataloader_hybrid = utils.DataLoader(
        dataset=utils.TensorDataset(
            torch.Tensor(val_signal_noisy),
            torch.Tensor(np.stack([val_label_bestfit, val_label_groundtruth], axis=2))),
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
                                              label_groundtruth=train_label_groundtruth,
                                              data_noisy=train_signal_noisy,
                                              label_bestfit=train_label_bestfit,
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
                                                label_groundtruth=val_label_groundtruth,
                                                data_noisy=val_signal_noisy,
                                                label_bestfit=val_label_bestfit,
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


def add_noise(noise_type, SNR, signal):
    """ Function to add noise to signals

            Inputs
            ------

            noise_type: string
                name of noise type

            SNR : int
                signal to noise ratio

            signal : ndarray
                noisefree signal

            Outputs
            -------
            signal_noisy : ndarray
                noisy signal

            Author: Sean C Epstein (https://seancepstein.github.io)
        """

    # number of independent signals
    n_signal = signal.shape[0]
    # number of sampling points per signal
    n_sampling = signal.shape[1]
    # standard deviation
    sigma = 1 / SNR
    # add rician noise
    if noise_type == 'rician':
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
        signal_noisy = signal + np.random.normal(scale=sigma, size=(n_signal, n_sampling))
    return signal_noisy


def traditional_fitting(model_name, signal, sampling_scheme, labels_groundtruth, SNR, noise_type,
                        seed_mean=True):
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

            Outputs
            -------
            labels_bestfit : ndarray
                best fit MLE parameters

            Author: Sean C Epstein (https://seancepstein.github.io)
        """
    # generate empty object to store bestfit parameters
    labels_bestfit = np.zeros_like(labels_groundtruth)
    # number of signals being fit
    n_signals = signal.shape[0]
    if model_name == 'ivim':
        if seed_mean:
            # determine fitting seed, set to mean parameter value
            seed = np.ones_like(labels_groundtruth) * labels_groundtruth.mean(axis=0)
        else:
            seed = labels_groundtruth

        # set upper and lower bounds for fitting
        bounds = Bounds(lb=np.array([0, 0, 0, 0]), ub=np.array([np.inf, 1, np.inf, 200]))
        # perform fitting
        for training_data in range(n_signals):
            # compute + save model parameters
            labels_bestfit[training_data, :] = minimize(fun=cost_gradient_descent, x0=seed[training_data, :],
                                                        bounds=bounds, args=[signal[training_data, :], SNR,
                                                                             model_name, noise_type, sampling_scheme]).x
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
    # define mapping from model name to generative functions
    generative_signal_handler = {
        'ivim': generate_IVIM
    }
    # define mapping from noise type to cost function
    objective_fn_handler = {
        'rician': rician_log_likelihood,
        'gaussian': gaussian_sse
    }
    # compute noisefree synthetic signal associated with parameter estimates
    synth_signal = generative_signal_handler[model_name](parameters, sampling_scheme)
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
                array of of noise-free signals synthesised from a generative model and proposed model parameters

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
    # return minus log likelihood (cost to minimise)
    return -log_likelihood


def train_general_network(script_dir, model_name, sampling_scheme, sampling_distribution, network_type, architecture,
                          n_nets, SNR, dataset_size, training_split, validation_split,
                          training_loader, validation_loader, method_name, validation_condition,
                          normalisation_factor, network_arch, noise_type, criterion, ignore_validation=False,
                          transfer_learning=False,state_dictionary=None):
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
        'NN_narrow': method_classes.netNarrow
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
            test_mle_uniform : method_classes.testResults
                object containing conventional MLE fits (using correct noise model) of the test data

            test_mle_uniform_approx : method_classes.testResults
                object containing conventional MLE fits (using wrong noise model) of the test data

            test_data_uniform : methdo_classes.testDataset
                object containing test data

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
        test_mle_uniform, test_data_uniform, test_mle_uniform_approx = analyse_test_data(n_repeats=n_repeats,
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
                                  'data/test/{}/{}/{}/test_data_uniform_{}_{}.pkl'.format(model_name,
                                                                                          noise_type,
                                                                                          sampling_distribution,
                                                                                          dataset_size,
                                                                                          SNR)), 'wb')
    pickle.dump(test_data_uniform, temp_file)
    temp_file.close()

    temp_file = open(os.path.join(script_dir,
                                  'data/test/{}/{}/{}/test_MLE_uniform_{}_{}.pkl'.format(model_name,
                                                                                         noise_type,
                                                                                         sampling_distribution,
                                                                                         dataset_size,
                                                                                         SNR)), 'wb')
    pickle.dump(test_mle_uniform, temp_file)
    temp_file.close()

    temp_file = open(os.path.join(script_dir,
                                  'data/test/{}/{}/{}/test_MLE_uniform_approx_{}_{}.pkl'.format(model_name,
                                                                                                noise_type,
                                                                                                sampling_distribution,
                                                                                                dataset_size,
                                                                                                SNR)), 'wb')
    pickle.dump(test_mle_uniform_approx, temp_file)
    temp_file.close()

    return test_mle_uniform, test_data_uniform, test_mle_uniform_approx


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
            mle_results : method_classes.testResults
                object containing conventional MLE fits (using correct noise model) of the test data

            mle_results_approx : method_classes.testResults
                object containing conventional MLE fits (using wrong noise model) of the test data

            test_data : methdo_classes.testDataset
                object containing test data

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

        # compute MLE estimates
        MLE_params = np.reshape(traditional_fitting(model_name=model_name,
                                                    signal=data_noisy_flat,
                                                    sampling_scheme=sampling_scheme,
                                                    labels_groundtruth=label_flat,
                                                    SNR=SNR,
                                                    noise_type=noise_type,
                                                    seed_mean=False),
                                (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats, 4))

        # compute corresponding signal
        MLE_signal = np.zeros(
            (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats, len(sampling_scheme)))

        for theta_0 in range(n_sampling[0]):
            for theta_1 in range(n_sampling[1]):
                for theta_2 in range(n_sampling[2]):
                    for theta_3 in range(n_sampling[3]):
                        for noise_repeat in range(n_repeats):
                            MLE_signal[theta_0, theta_1, theta_2, theta_3, noise_repeat, :] = generative_dictionary[
                                model_name](
                                [MLE_params[theta_0, theta_1, theta_2, theta_3, noise_repeat, 0],
                                 MLE_params[theta_0, theta_1, theta_2, theta_3, noise_repeat, 1],
                                 MLE_params[theta_0, theta_1, theta_2, theta_3, noise_repeat, 2],
                                 MLE_params[theta_0, theta_1, theta_2, theta_3, noise_repeat, 3]],
                                sampling_scheme)

        # calculate signal SSE
        MLE_sse = np.sum((data_noisy - MLE_signal) ** 2, axis=5)
        MLE_sse_mean = np.mean(MLE_sse, axis=4)
        # caculate mean MLE param
        MLE_params_mean = np.mean(MLE_params, axis=4)
        # calculate MLE standard deviation under noise
        MLE_params_std = np.std(MLE_params, axis=4)

        MLE_params_flat = np.reshape(np.transpose(MLE_params, (0, 1, 2, 3, 5, 4)),
                                     (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4, n_repeats))
        groundtruth_params_flat = matlib.repmat(
            np.reshape(label, (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4, 1)), 1, n_repeats)

        MLE_params_rmse = np.reshape(np.mean(np.sqrt((MLE_params_flat - groundtruth_params_flat) ** 2), axis=1),
                                     (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], 4))

        # compute MLE estimates
        MLE_params_approx = np.reshape(traditional_fitting(model_name=model_name,
                                                           signal=data_noisy_flat,
                                                           sampling_scheme=sampling_scheme,
                                                           labels_groundtruth=label_flat,
                                                           SNR=SNR,
                                                           noise_type='gaussian',
                                                           seed_mean=False),
                                       (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats, 4), )

        MLE_signal_approx = np.zeros(
            (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats, len(sampling_scheme)))

        for theta_0 in range(n_sampling[0]):
            for theta_1 in range(n_sampling[1]):
                for theta_2 in range(n_sampling[2]):
                    for theta_3 in range(n_sampling[3]):
                        for noise_repeat in range(n_repeats):
                            MLE_signal_approx[theta_0, theta_1, theta_2, theta_3, noise_repeat, :] = \
                                generative_dictionary[
                                    model_name](
                                    [MLE_params_approx[theta_0, theta_1, theta_2, theta_3, noise_repeat, 0],
                                     MLE_params_approx[theta_0, theta_1, theta_2, theta_3, noise_repeat, 1],
                                     MLE_params_approx[theta_0, theta_1, theta_2, theta_3, noise_repeat, 2],
                                     MLE_params_approx[theta_0, theta_1, theta_2, theta_3, noise_repeat, 3]],
                                    sampling_scheme)

        MLE_sse_approx = np.sum((data_noisy - MLE_signal_approx) ** 2, axis=5)
        MLE_sse_approx_mean = np.mean(MLE_sse_approx, axis=4)
        MLE_params_approx_mean = np.mean(MLE_params_approx, axis=4)
        MLE_params_approx_std = np.std(MLE_params_approx, axis=4)

        MLE_params_approx_flat = np.reshape(np.transpose(MLE_params_approx, (0, 1, 2, 3, 5, 4)),
                                            (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4,
                                             n_repeats))
        groundtruth_params_flat = matlib.repmat(
            np.reshape(label, (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4, 1)), 1, n_repeats)

        MLE_params_approx_rmse = np.reshape(
            np.mean(np.sqrt((MLE_params_approx_flat - groundtruth_params_flat) ** 2), axis=1),
            (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], 4))

    mle_results = method_classes.testResults(network_object=None,
                                             test_data=data_noisy,
                                             param_predictions=MLE_params,
                                             param_predictions_mean=MLE_params_mean,
                                             param_predictions_std=MLE_params_std,
                                             signal_predictions_sse=MLE_sse,
                                             signal_predictions_sse_mean=MLE_sse_mean,
                                             param_predictions_rmse=MLE_params_rmse)

    mle_results_approx = method_classes.testResults(network_object=None,
                                                    test_data=data_noisy,
                                                    param_predictions=MLE_params_approx,
                                                    param_predictions_mean=MLE_params_approx_mean,
                                                    param_predictions_std=MLE_params_approx_std,
                                                    signal_predictions_sse=MLE_sse_approx,
                                                    signal_predictions_sse_mean=MLE_sse_approx_mean,
                                                    param_predictions_rmse=MLE_params_approx_rmse)

    test_data = method_classes.testDataset(test_data=data_noisefree,
                                           test_data_flat=data_noisefree_flat,
                                           test_label=label,
                                           test_label_flat=None,
                                           test_data_noisy=data_noisy,
                                           test_data_noisy_flat=data_noisy_flat,
                                           n_repeats=n_repeats,
                                           n_sampling=n_sampling,
                                           extent_scaling=extent_scaling)

    return mle_results, test_data, mle_results_approx


def test_network(model_name, test_data, test_label, n_sampling, n_repeats,
                 mle_flag=False, mle_data=None, network=None):
    """ Function to test network performance

            Inputs
            ------

            model_name : string
                name of signal model being fit

            test_data : methdo_classes.testDataset
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

            Outputs
            -------
            mle_results : method_classes.testResults
                object containing conventional MLE fits (using correct noise model) of the test data

            mle_results_approx : method_classes.testResults
                object containing conventional MLE fits (using wrong noise model) of the test data

            test_data : methdo_classes.testDataset
                object containing test data

            Author: Sean C Epstein (https://seancepstein.github.io)
        """

    # if assessing conventional fit, extract parameters from MLE data
    if mle_flag:
        fitted_parameters = mle_data.param_predictions
        bestfit_sse = mle_data.signal_predictions_sse
    else:
        # set network to evaluation mode
        network.eval
        # obtain network predictions
        with torch.no_grad():
            fitted_signal, fitted_parameters = network(torch.from_numpy(test_data.astype(np.float32)))
        bestfit_sse = np.sum((test_data - fitted_signal.numpy()) ** 2, axis=1)

    if model_name == 'ivim':

        if mle_flag:
            pass
        else:
            # sum of squared errors of signal associated with best-fit parameters
            bestfit_sse = np.reshape(bestfit_sse,
                                     (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], n_repeats))
            # reshaped best-fit parameters
            fitted_parameters = np.reshape(fitted_parameters.numpy(),
                                           (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3],
                                            n_repeats, 4))

        # mean sum of squared errors of signal associated with best-fit parameters across noise
        bestfit_sse_mean = np.mean(bestfit_sse, axis=4)

        #  mean best-fit parameters under noise
        fitted_parameters_mean = np.mean(fitted_parameters, axis=4)
        # standard deviation of best-fit parameters under noise
        fitted_parameters_std = np.std(fitted_parameters, axis=4)

        # reshaped groundtruth parameters, used to compute rmse
        test_label_flat = matlib.repmat(
            np.reshape(test_label, (n_sampling[0] * n_sampling[1] * n_sampling[2] * n_sampling[3] * 4, 1)), 1,
            n_repeats)
        test_label_expanded = np.swapaxes(
            np.reshape(test_label_flat, (n_sampling[0], n_sampling[1], n_sampling[2], n_sampling[3], 4, n_repeats)),
            4, 5)

        # compute rmse of best-fit parameters
        fitted_parameters_rmse = np.sqrt((fitted_parameters - test_label_expanded) ** 2)
    else:
        sys.exit("Implement other signal models here")

    return method_classes.testResults(network_object=network,
                                      test_data=test_data,
                                      param_predictions=fitted_parameters,
                                      param_predictions_mean=fitted_parameters_mean,
                                      param_predictions_std=fitted_parameters_std,
                                      signal_predictions_sse=bestfit_sse,
                                      signal_predictions_sse_mean=bestfit_sse_mean,
                                      param_predictions_rmse=fitted_parameters_rmse)