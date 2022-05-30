import torch
import torch.nn as nn
import method_functions


class Dataset:
    def __init__(self, model_name, parameter_distribution, parameter_range, sampling_scheme, noise_type, SNR, n_signals,
                 batch_size, num_workers, data_clean, label_groundtruth, data_noisy, label_bestfit,
                 dataloader_supervised_groundtruth, dataloader_supervised_mle, dataloader_supervised_mle_approx,
                 dataloader_hybrid, train_split, val_split, sampling_distribution):
        self.val_split = val_split
        self.train_split = train_split
        self.n_signals = n_signals
        self.num_workers = num_workers
        self.SNR = SNR
        self.noise_type = noise_type
        self.sampling_scheme = sampling_scheme
        self.parameter_range = parameter_range
        self.parameter_distribution = parameter_distribution
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_clean = data_clean
        self.label_groundtruth = label_groundtruth
        self.data_noisy = data_noisy
        self.label_bestfit = label_bestfit
        self.dataloader_supervised_groundtruth = dataloader_supervised_groundtruth
        self.dataloader_supervised_mle = dataloader_supervised_mle
        self.dataloader_supervised_mle_approx = dataloader_supervised_mle_approx
        self.sampling_distribution = sampling_distribution
        self.dataloader_hybrid = dataloader_hybrid


class trainedNet:
    def __init__(self, best_network, network_object, best_network_idx, training_loss, validation_loss,
                 last_epoch_tracker, network_type, architecture, n_nets,
                 sorted_networks_idx, batch_loss, network_tracker):
        self.n_nets = n_nets
        self.architecture = architecture
        self.network_type = network_type
        self.best_network = best_network
        self.network_object = network_object
        self.best_network_idx = best_network_idx
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        self.last_epoch_tracker = last_epoch_tracker
        self.sorted_networks_idx = sorted_networks_idx
        self.batch_loss = batch_loss
        self.network_tracker = network_tracker


class netNarrow(nn.Module):
    def __init__(self, sampling_scheme, n_params, model_name):
        super(netNarrow, self).__init__()
        self.model_name = model_name
        # independent variables sampling scheme
        self.sampling_scheme = torch.FloatTensor(sampling_scheme)

        # set up 3 fully connected hidden layers
        self.fc_layers = nn.ModuleList()
        for i in range(3):
            self.fc_layers.extend([nn.Linear(len(sampling_scheme), len(sampling_scheme)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(sampling_scheme), n_params))

    def forward(self, x):
        # convert outputs into signal
        generative_dictionary = {
            'ivim': method_functions.generate_IVIM_tensor
        }

        params = torch.abs(self.encoder(x))
        signal = generative_dictionary[self.model_name](params, self.sampling_scheme)

        return signal, params


class testDataset:
    def __init__(self, test_data, test_data_flat, test_label, test_label_flat, test_data_noisy, test_data_noisy_flat,
                 n_repeats, n_sampling, extent_scaling):
        self.test_data = test_data
        self.test_data_flat = test_data_flat
        self.test_label = test_label
        self.test_label_flat = test_label_flat
        self.test_data_noisy = test_data_noisy
        self.test_data_noisy_flat = test_data_noisy_flat
        self.n_repeats = n_repeats
        self.n_sampling = n_sampling
        self.extent_scaling = extent_scaling


class testResults:
    def __init__(self, network_object, test_data, param_predictions, param_predictions_mean, param_predictions_std,
                 signal_predictions_sse, signal_predictions_sse_mean, param_predictions_rmse):
        self.network_object = network_object
        self.test_data = test_data
        self.param_predictions = param_predictions
        self.param_predictions_mean = param_predictions_mean
        self.param_predictions_std = param_predictions_std
        self.signal_predictions_sse = signal_predictions_sse
        self.signal_predictions_sse_mean = signal_predictions_sse_mean
        self.param_predictions_rmse = param_predictions_rmse
