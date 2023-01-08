import matplotlib.pyplot as plt
import numpy.matlib as matlib
import nibabel, matplotlib, os
import numpy as np
import matplotlib.colors as colors
from matplotlib.patches import Rectangle


def generate_plots(save_plots, network_arch, dataset_size, training_data, test_data, SNR, script_dir,
                   parameter_range, clinical_groundtruth_masked, test_mle_mean,
                   test_supervised_groundtruth, test_supervised_mle, test_supervised_mle_approx, test_selfsupervised,
                   test_supervised_mle_weighted_f, test_hybrid,
                   test_supervised_mle_weighted_Dslow, test_supervised_mle_weighted_Dfast,
                   test_clinical_real_mle_mean, test_clinical_real_supervised_mle, test_clinical_real_selfsupervised,
                   test_clinical_synth_mle_mean, test_clinical_synth_supervised_mle, test_clinical_synth_selfsupervised,
                   test_clinical_synth_supervised_mle_approx, test_clinical_synth_supervised_groundtruth,
                   test_clinical_real_supervised_mle_approx, test_clinical_real_supervised_groundtruth,
                   fig_2=False, fig_3=False, fig_4=False, fig_5=False, fig_6=False, fig_7=False, fig_8=False,
                   fig_9=False, fig_10=False, fig_11=False, fig_12=False):
    # Replicate published plots
    # comparison between all methods, uniform test data
    if (fig_2 and SNR == 15) or (fig_3 and SNR == 30) or (fig_2 and SNR == 20):
        figure_23, ax_23 = plot_summary_uniform(test_data=test_data,
                                                n_networks=5,
                                                network_data=[test_mle_mean,
                                                              test_supervised_mle,
                                                              test_supervised_mle_approx,
                                                              test_selfsupervised,
                                                              test_supervised_groundtruth],
                                                label=[r'$Conventional MLE$',
                                                       r'$Supervised_{MLE,\,Rician}$',
                                                       r'$Supervised_{MLE,\,Gaussian}$',
                                                       r'$Self\mathrm{-}supervised$',
                                                       r'$Supervised_{Groundtruth}$'],
                                                color=['k',
                                                       'r',
                                                       'r',
                                                       'r',
                                                       'b'],
                                                linestyle=['solid',
                                                           'solid',
                                                           ':',
                                                           'dashed',
                                                           'solid'])

        if save_plots:
            if SNR == 15 or SNR == 20:
                figure_23.savefig(
                    os.path.join(script_dir,
                                 'figures/{}/{}/{}/{}/figure_2_{}_{}.pdf'.format(network_arch,
                                                                                 training_data.model_name,
                                                                                 training_data.noise_type,
                                                                                 training_data.sampling_distribution,
                                                                                 dataset_size,
                                                                                 training_data.SNR)))
            elif SNR == 30:
                figure_23.savefig(
                    os.path.join(script_dir,
                                 'figures/{}/{}/{}/{}/figure_3_{}_{}.pdf'.format(network_arch,
                                                                                 training_data.model_name,
                                                                                 training_data.noise_type,
                                                                                 training_data.sampling_distribution,
                                                                                 dataset_size,
                                                                                 training_data.SNR)))
            plt.close(figure_23)
        else:
            figure_23.show()

    # supervised_gt vs conventional MLE, quiverplot
    if fig_4 and SNR == 15:

        figure_4, ax_4 = plot_quiver_two_method_summary(test_results_1=test_supervised_groundtruth,
                                                        test_results_2=test_mle_mean,
                                                        test_groundtruth=test_data,
                                                        name_1=r'$Supervised_{Groundtruth}$',
                                                        name_2=r'$Conventional MLE$', training_data=training_data)

        figure_4.tight_layout()

        if save_plots:
            figure_4.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_4_{}_{}.pdf'.format(network_arch,
                                                                             training_data.model_name,
                                                                             training_data.noise_type,
                                                                             training_data.sampling_distribution,
                                                                             dataset_size,
                                                                             training_data.SNR)))
            plt.close(figure_4)
        else:
            figure_4.show()

    # comparison between all methods, real clinical test data
    if fig_5 and SNR == 15:
        figure_5, ax_5 = plot_summary_clinical(
            test_label=clinical_groundtruth_masked,
            n_networks=5,
            network_data=[test_clinical_real_mle_mean,
                          test_clinical_real_supervised_mle,
                          test_clinical_real_selfsupervised,
                          test_clinical_real_supervised_mle_approx,
                          test_clinical_real_supervised_groundtruth],
            label=[r'$Conventional MLE$',
                   r'$Supervised_{MLE,\,Rician}$',
                   r'$Self\mathrm{-}supervised$',
                   r'$Supervised_{MLE,\,Gaussian}$',
                   r'$Supervised_{Groundtruth}$'],
            color=['k',
                   'r',
                   'r',
                   'r',
                   'b'],
            linestyle=['solid',
                       'solid',
                       'dashed',
                       ':',
                       'solid'],
            parameter_range=parameter_range,
            filter=True)

        if save_plots:
            figure_5.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_5_{}_{}.pdf'.format(network_arch,
                                                                             training_data.model_name,
                                                                             training_data.noise_type,
                                                                             training_data.sampling_distribution,
                                                                             dataset_size,
                                                                             training_data.SNR)))
            plt.close(figure_5)

    # comparison between all methods, synthetic clinical test data
    if fig_6 and SNR == 15:
        figure_6, ax_6 = plot_summary_clinical(
            test_label=clinical_groundtruth_masked,
            n_networks=5,
            network_data=[test_clinical_synth_mle_mean,
                          test_clinical_synth_supervised_mle,
                          test_clinical_synth_selfsupervised,
                          test_clinical_synth_supervised_mle_approx,
                          test_clinical_synth_supervised_groundtruth],
            label=[r'$Conventional MLE$',
                   r'$Supervised_{MLE,\,Rician}$',
                   r'$Self\mathrm{-}supervised$',
                   r'$Supervised_{MLE,\,Gaussian}$',
                   r'$Supervised_{Groundtruth}$'],
            color=['k',
                   'r',
                   'r',
                   'r',
                   'b'],
            linestyle=['solid',
                       'solid',
                       'dashed',
                       ':',
                       'solid'],
            parameter_range=parameter_range,
            filter=True)

        if save_plots:
            figure_6.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_6_{}_{}.pdf'.format(network_arch,
                                                                             training_data.model_name,
                                                                             training_data.noise_type,
                                                                             training_data.sampling_distribution,
                                                                             dataset_size,
                                                                             training_data.SNR)))
            plt.close(figure_6)

    # exemplar invivo parameter maps
    if fig_7 and SNR == 15:
        mask_no_bowel = nibabel.load(os.path.join(script_dir, 'clinical_data/epstein_bowel_mask.nii')).get_fdata()

        figure_7, ax_7 = plot_parameter_maps_clinical(
            z_idx=4,
            mask=mask_no_bowel,
            groundtruth_flat=clinical_groundtruth_masked,
            volumes=[test_clinical_real_mle_mean.param_predictions,
                     test_clinical_real_supervised_mle.param_predictions,
                     test_clinical_real_selfsupervised.param_predictions,
                     test_clinical_real_supervised_groundtruth.param_predictions],
            labels=['Conventional MLE\n',
                    'Supervised (MLE)\n',
                    'Self-supervised\n',
                    'Supervised (groundtruth)\n'],
            parameter_idx=2)
        if save_plots:
            figure_7.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_7_{}_{}.pdf'.format(network_arch,
                                                                             training_data.model_name,
                                                                             training_data.noise_type,
                                                                             training_data.sampling_distribution,
                                                                             dataset_size,
                                                                             training_data.SNR)))
            plt.close(figure_7)

    # comparison between supervised_mle, and individual-parameter-weighted versions of this method
    if fig_8 and SNR == 15:
        figure_8, ax_8 = plot_summary_uniform_weighted(test_data=test_data,
                                                       reference_network=test_supervised_mle,
                                                       weighted_network=[test_supervised_mle_weighted_f,
                                                                         test_supervised_mle_weighted_Dslow,
                                                                         test_supervised_mle_weighted_Dfast],
                                                       label=['$Supervised_{MLE, Rician}$',
                                                              'Weighted $Supervised_{MLE, Rician}$'],
                                                       color=['k', 'r'],
                                                       linestyle=['dashdot', 'solid'])

        if save_plots:
            figure_8.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_8_{}_{}.pdf'.format(network_arch,
                                                                             training_data.model_name,
                                                                             training_data.noise_type,
                                                                             training_data.sampling_distribution,
                                                                             dataset_size,
                                                                             training_data.SNR)))
            plt.close(figure_8)
        else:
            figure_8.show()

    # quiverplot comparison between supervised_mle_rician and supervised_mle_gaussian
    if fig_9 and SNR == 15:
        figure_9, ax_9 = plot_quiver_two_method_focus(test_results_1=test_supervised_mle_approx,
                                                      test_results_2=test_supervised_mle,
                                                      test_groundtruth=test_data,
                                                      name_1=r'$Supervised_{MLE, Gaussian}$',
                                                      name_2=r'$Supervised_{MLE, Rician}$',
                                                      training_data=training_data)

        figure_9.tight_layout()

        if save_plots:
            figure_9.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_9_{}_{}.pdf'.format(network_arch,
                                                                             training_data.model_name,
                                                                             training_data.noise_type,
                                                                             training_data.sampling_distribution,
                                                                             dataset_size,
                                                                             training_data.SNR)))
            plt.close(figure_9)
        else:
            figure_9.show()

    # comparison between supervised_mle, supervised_groundtruth, and a hybrid mix of the two
    if fig_10 and SNR == 15:

        figure_10, ax_10 = plot_summary_uniform(test_data=test_data,
                                                n_networks=3,
                                                network_data=[test_supervised_mle,
                                                              test_supervised_groundtruth,
                                                              test_hybrid],
                                                label=['$Supervised_{MLE, Rician}$',
                                                       '$Supervised_{GT}$',
                                                       '$Hybrid$'],
                                                color=['k', 'k', 'r'],
                                                linestyle=['dashdot', 'solid', 'dashed'])

        if save_plots:
            figure_10.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_10_{}_{}.pdf'.format(network_arch,
                                                                              training_data.model_name,
                                                                              training_data.noise_type,
                                                                              training_data.sampling_distribution,
                                                                              dataset_size,
                                                                              training_data.SNR)))
            plt.close(figure_10)
        else:
            figure_10.show()

    # marginalisation sanity check #1
    if fig_11 and SNR == 15:
        figure_11, ax_11 = plot_2D_distribution_comparison(test_data=test_data,
                                                           network_1_data=test_supervised_groundtruth,
                                                           network_2_data=test_supervised_mle,
                                                           label_1='$Supervised_{Groundtruth}$',
                                                           label_2='$Supervised_{MLE, Rician}$')

        if save_plots:
            figure_11.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_11_{}_{}.pdf'.format(
                                 network_arch,
                                 training_data.model_name,
                                 training_data.noise_type,
                                 training_data.sampling_distribution,
                                 dataset_size,
                                 training_data.SNR)))
            plt.close(figure_11)
        else:
            figure_11.show()

    # marginalisation sanity check #2
    if fig_12 and SNR == 15:
        figure_12, ax_12 = plot_two_method_fixedpoint_comparison(results_1=test_supervised_groundtruth,
                                                                 results_2=test_supervised_mle,
                                                                 test_data=test_data,
                                                                 title='$Supervised_{Groundtruth}$ - $Supervised_{MLE, Rician}$',
                                                                 parameter_range=training_data.parameter_range,
                                                                 SNR=training_data.SNR,
                                                                 label=[r'$Supervised_{Groundtruth}$',
                                                                        r'$Supervised_{MLE,\,Rician}$'],
                                                                 color=['b',
                                                                        'r'],
                                                                 linestyle=['solid', 'solid'],
                                                                 d_slow_idx=(1, 8))
        if save_plots:
            figure_12.savefig(
                os.path.join(script_dir,
                             'figures/{}/{}/{}/{}/figure_12_{}_{}.pdf'.format(
                                 network_arch,
                                 training_data.model_name,
                                 training_data.noise_type,
                                 training_data.sampling_distribution,
                                 dataset_size,
                                 training_data.SNR)))
            plt.close(figure_12)
        else:
            figure_12.show()

    # fig_13 = True
    # if fig_13 and SNR == 20:
    #     figure_13, ax_13 = plot_performance_maps(results=[test_mle_mean,
    #                                                       test_supervised_mle,
    #                                                       test_supervised_groundtruth,
    #                                                       test_selfsupervised,
    #                                                       test_supervised_mle_filtered_weighted_Dslow,
    #                                                       test_supervised_mle_segmented_weighted_Dslow],
    #                                              test_data=test_data,
    #                                              title=[r'$Conventional MLE$',
    #                                                     r'$Supervised_{MLE,\,Rician}$',
    #                                                     r'$Supervised_{Groundtruth}$',
    #                                                     r'$Self\mathrm{-}supervised$',
    #                                                     r'$Supervised_{MLE,\,Rician, filtered}$',
    #                                                     r'$Supervised_{MLE,\,Rician, segmented}$'],
    #                                              parameter_range=training_data.parameter_range,
    #                                              SNR=SNR,
    #                                              n_method=6)
    #
    #     if save_plots:
    #         figure_13.savefig(
    #             os.path.join(script_dir,
    #                          'figures/{}/{}/{}/{}/figure_13_{}_{}.pdf'.format(
    #                              network_arch,
    #                              training_data.model_name,
    #                              training_data.noise_type,
    #                              training_data.sampling_distribution,
    #                              dataset_size,
    #                              training_data.SNR)))
    #         plt.close(figure_13)
    #
    #     figure_13_marg_dslow, ax_13_marg_dslow = plot_performance_maps_marginalised_dslow(results=[test_mle_mean,
    #                                                                              test_supervised_mle,
    #                                                                              test_supervised_groundtruth,
    #                                                                              test_selfsupervised,
    #                                                                              test_supervised_mle_filtered,
    #                                                                              test_supervised_mle_segmented],
    #                                                                     test_data=test_data,
    #                                                                     title=[r'$Conventional MLE$',
    #                                                                            r'$Supervised_{MLE,\,Rician}$',
    #                                                                            r'$Supervised_{Groundtruth}$',
    #                                                                            r'$Self\mathrm{-}supervised$',
    #                                                                            r'$Supervised_{MLE,\,Rician, filtered}$',
    #                                                                            r'$Supervised_{MLE,\,Rician, segmented}$'],
    #                                                                     parameter_range=training_data.parameter_range,
    #                                                                     SNR=SNR,
    #                                                                     n_method=6)
    #
    #     if save_plots:
    #         figure_13_marg_dslow.savefig(
    #             os.path.join(script_dir,
    #                          'figures/{}/{}/{}/{}/figure_13_marg_dslow_{}_{}.pdf'.format(
    #                              network_arch,
    #                              training_data.model_name,
    #                              training_data.noise_type,
    #                              training_data.sampling_distribution,
    #                              dataset_size,
    #                              training_data.SNR)))
    #         plt.close(figure_13_marg_dslow)
    #
    #     figure_13_marg_f, ax_13_marg_f = plot_performance_maps_marginalised_f(results=[test_mle_mean,
    #                                                                                    test_supervised_mle,
    #                                                                                    test_supervised_groundtruth,
    #                                                                                    test_selfsupervised,
    #                                                                                    test_supervised_mle_filtered,
    #                                                                                    test_supervised_mle_segmented],
    #                                                                           test_data=test_data,
    #                                                                           title=[r'$Conventional MLE$',
    #                                                                                  r'$Supervised_{MLE,\,Rician}$',
    #                                                                                  r'$Supervised_{Groundtruth}$',
    #                                                                                  r'$Self\mathrm{-}supervised$',
    #                                                                                  r'$Supervised_{MLE,\,Rician, filtered}$',
    #                                                                                  r'$Supervised_{MLE,\,Rician, segmented}$'],
    #                                                                           parameter_range=training_data.parameter_range,
    #                                                                           SNR=SNR,
    #                                                                           n_method=6)
    #
    #     if save_plots:
    #         figure_13_marg_f.savefig(
    #             os.path.join(script_dir,
    #                          'figures/{}/{}/{}/{}/figure_13_marg_f_{}_{}.pdf'.format(
    #                              network_arch,
    #                              training_data.model_name,
    #                              training_data.noise_type,
    #                              training_data.sampling_distribution,
    #                              dataset_size,
    #                              training_data.SNR)))
    #         plt.close(figure_13_marg_f)

    return


def plot_quiver_two_method_summary(test_results_1, test_results_2, test_groundtruth, name_1, name_2,
                                   training_data, extent_scaling=0.25):
    fig_quiver, ax_quiver = plt.subplots(2, 3, figsize=(35, 25))
    ax_quiver = ax_quiver.ravel()

    fig_quiver, ax_quiver[0], _ = plot_quiver(figure=fig_quiver,
                                              axes=ax_quiver[0],
                                              groundtruth=test_groundtruth.test_label[0, 0, :, :, [2, 3]],
                                              predicted_mean=np.transpose(
                                                  np.mean(test_results_1.param_predictions,
                                                          axis=(0, 1, 4))[:, :, [2, 3]], axes=(2, 0, 1)),
                                              title=name_1 + ', $f$ marginalisation',
                                              parameter_range=training_data.parameter_range[[2, 3], :],
                                              extent_scaling=extent_scaling,
                                              x_axis_label='$D_{slow}$',
                                              y_axis_label='$D_{fast}$')
    fig_quiver, ax_quiver[1], _ = plot_quiver(figure=fig_quiver,
                                              axes=ax_quiver[1],
                                              groundtruth=test_groundtruth.test_label[0, :, 0, :, [1, 3]],
                                              predicted_mean=np.transpose(
                                                  np.mean(test_results_1.param_predictions,
                                                          axis=(0, 2, 4))[:, :, [1, 3]], axes=(2, 0, 1)),
                                              title=name_1 + ', $D_{slow}$ marginalisation',
                                              parameter_range=training_data.parameter_range[[1, 3], :],
                                              extent_scaling=extent_scaling,
                                              x_axis_label='$f$',
                                              y_axis_label='$D_{fast}$')
    fig_quiver, ax_quiver[2], _ = plot_quiver(figure=fig_quiver,
                                              axes=ax_quiver[2],
                                              groundtruth=test_groundtruth.test_label[0, :, :, 0, [1, 2]],
                                              predicted_mean=np.transpose(
                                                  np.mean(test_results_1.param_predictions,
                                                          axis=(0, 3, 4))[:, :, [1, 2]], axes=(2, 0, 1)),
                                              title=name_1 + ', $D_{fast}$ marginalisation',
                                              parameter_range=training_data.parameter_range[[1, 2], :],
                                              extent_scaling=extent_scaling,
                                              x_axis_label='$f$',
                                              y_axis_label='$D_{slow}$')

    fig_quiver, ax_quiver[3], _ = plot_quiver(figure=fig_quiver,
                                              axes=ax_quiver[3],
                                              groundtruth=test_groundtruth.test_label[0, 0, :, :, [2, 3]],
                                              predicted_mean=np.transpose(
                                                  np.mean(test_results_2.param_predictions,
                                                          axis=(0, 1, 4))[:, :, [2, 3]], axes=(2, 0, 1)),
                                              title=name_2 + ', $f$ marginalisation',
                                              parameter_range=training_data.parameter_range[[2, 3], :],
                                              extent_scaling=extent_scaling,
                                              x_axis_label='$D_{slow}$',
                                              y_axis_label='$D_{fast}$')
    fig_quiver, ax_quiver[4], _ = plot_quiver(figure=fig_quiver,
                                              axes=ax_quiver[4],
                                              groundtruth=test_groundtruth.test_label[0, :, 0, :, [1, 3]],
                                              predicted_mean=np.transpose(
                                                  np.mean(test_results_2.param_predictions,
                                                          axis=(0, 2, 4))[:, :, [1, 3]], axes=(2, 0, 1)),
                                              title=name_2 + ', $D_{slow}$ marginalisation',
                                              parameter_range=training_data.parameter_range[[1, 3], :],
                                              extent_scaling=extent_scaling,
                                              x_axis_label='$f$',
                                              y_axis_label='$D_{fast}$')
    fig_quiver, ax_quiver[5], _ = plot_quiver(figure=fig_quiver,
                                              axes=ax_quiver[5],
                                              groundtruth=test_groundtruth.test_label[0, :, :, 0, [1, 2]],
                                              predicted_mean=np.transpose(
                                                  np.mean(test_results_2.param_predictions,
                                                          axis=(0, 3, 4))[:, :, [1, 2]], axes=(2, 0, 1)),
                                              title=name_2 + ', $D_{fast}$ marginalisation',
                                              parameter_range=training_data.parameter_range[[1, 2], :],
                                              extent_scaling=extent_scaling,
                                              x_axis_label='$f$',
                                              y_axis_label='$D_{slow}$')

    return fig_quiver, ax_quiver


def plot_quiver_two_method_focus(test_results_1, test_results_2, test_groundtruth, name_1, name_2,
                                 training_data, extent_scaling=0.25):
    fig_quiver, ax_quiver = plt.subplots(1, 2, figsize=(25, 15))
    ax_quiver = ax_quiver.ravel()

    fig_quiver, ax_quiver[0], _ = plot_quiver(figure=fig_quiver,
                                              axes=ax_quiver[0],
                                              groundtruth=test_groundtruth.test_label[0, :, :, 0, [1, 2]],
                                              predicted_mean=np.transpose(
                                                  np.mean(test_results_1.param_predictions,
                                                          axis=(0, 3, 4))[:, :, [1, 2]], axes=(2, 0, 1)),
                                              title=name_1 + ', $D_{fast}$ marginalisation',
                                              parameter_range=training_data.parameter_range[[1, 2], :],
                                              extent_scaling=extent_scaling,
                                              x_axis_label='$f$',
                                              y_axis_label='$D_{slow}$')

    fig_quiver, ax_quiver[1], _ = plot_quiver(figure=fig_quiver,
                                              axes=ax_quiver[1],
                                              groundtruth=test_groundtruth.test_label[0, :, :, 0, [1, 2]],
                                              predicted_mean=np.transpose(
                                                  np.mean(test_results_2.param_predictions,
                                                          axis=(0, 3, 4))[:, :, [1, 2]], axes=(2, 0, 1)),
                                              title=name_2 + ', $D_{fast}$ marginalisation',
                                              parameter_range=training_data.parameter_range[[1, 2], :],
                                              extent_scaling=extent_scaling,
                                              x_axis_label='$f$',
                                              y_axis_label='$D_{slow}$')

    return fig_quiver, ax_quiver


def plot_summary_uniform(test_data, network_data, label, color, linestyle, n_networks):
    figure, axes = plt.subplots(3, 3, figsize=(35, 25))
    axes = axes.ravel()
    linewidth = 4

    for net_idx in range(n_networks):
        axes[0].plot(
            test_data.test_label[0, :, 0, 0, 1],
            (np.mean(network_data[net_idx].param_predictions_mean[:, :, :, :], axis=(0, 2, 3))[:,
             1] - test_data.test_label[0, :, 0, 0, 1]),
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
        axes[1].plot(
            test_data.test_label[0, 0, :, 0, 2],
            (np.mean(network_data[net_idx].param_predictions_mean[:, :, :, :], axis=(0, 1, 3))[:,
             2] - test_data.test_label[0, 0, :, 0, 2]),
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
        axes[2].plot(
            test_data.test_label[0, 0, 0, :, 3],
            (np.mean(network_data[net_idx].param_predictions_mean[:, :, :, :], axis=(0, 1, 2))[:,
             3] - test_data.test_label[0, 0, 0, :, 3]),
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
        axes[3].plot(
            test_data.test_label[0, :, 0, 0, 1],
            np.mean((network_data[net_idx].param_predictions_std[:, :, :, :]), axis=(0, 2, 3))[:, 1],
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
        axes[4].plot(
            test_data.test_label[0, 0, :, 0, 2],
            np.mean((network_data[net_idx].param_predictions_std[:, :, :, :]), axis=(0, 1, 3))[:, 2],
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
        axes[5].plot(
            test_data.test_label[0, 0, 0, :, 3],
            np.mean((network_data[net_idx].param_predictions_std[:, :, :, :]), axis=(0, 1, 2))[:, 3],
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
        axes[6].errorbar(
            test_data.test_label[0, :, 0, 0, 1],
            np.mean(network_data[net_idx].param_predictions_rmse[:, :, :, :], axis=(0, 2, 3))[:, 1],
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
        axes[7].errorbar(
            test_data.test_label[0, 0, :, 0, 2],
            np.mean(network_data[net_idx].param_predictions_rmse[:, :, :, :], axis=(0, 1, 3))[:, 2],
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
        axes[8].errorbar(
            test_data.test_label[0, 0, 0, :, 3],
            np.mean(network_data[net_idx].param_predictions_rmse[:, :, :, :], axis=(0, 1, 2))[:, 3],
            label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])

    axes[0].set_title('$f$', fontsize=30)
    axes[1].set_title('$D_{slow}$', fontsize=30)
    axes[2].set_title('$D_{fast}$', fontsize=30)
    axes[0].legend(ncol=2)

    for axes_idx in range(3):
        axes[axes_idx].set_ylabel('Bias')
        axes[axes_idx].axhline(y=0, color='k')
        axes[axes_idx + 3].set_ylim(bottom=0)
        axes[axes_idx + 3].set_ylabel('Standard deviation')
        axes[axes_idx + 6].set_ylabel('RMSE')
        axes[axes_idx + 6].set_xlabel('Parameter value')
        axes[axes_idx * 3].set_xlim(
            (min(test_data.test_label[0, :, 0, 0, 1]), max(test_data.test_label[0, :, 0, 0, 1])))
        axes[axes_idx * 3 + 1].set_xlim(
            (min(test_data.test_label[0, 0, :, 0, 2]), max(test_data.test_label[0, 0, :, 0, 2])))
        axes[axes_idx * 3 + 2].set_xlim(
            (min(test_data.test_label[0, 0, 0, :, 3]), max(test_data.test_label[0, 0, 0, :, 3])))

    ylim_range_f = []
    ylim_range_Dslow = []
    ylim_range_Dfast = []

    for plot_idx in range(3):
        ylim_range_f.append(axes[plot_idx * 3].get_ylim()[1] - axes[plot_idx * 3].get_ylim()[0])
        ylim_range_Dslow.append(axes[plot_idx * 3 + 1].get_ylim()[1] - axes[plot_idx * 3 + 1].get_ylim()[0])
        ylim_range_Dfast.append(axes[plot_idx * 3 + 2].get_ylim()[1] - axes[plot_idx * 3 + 2].get_ylim()[0])

    ylim_f_max = 1.1 * max(ylim_range_f)
    ylim_Dslow_max = 1.1 * max(ylim_range_Dslow)
    ylim_Dfast_max = 1.1 * max(ylim_range_Dfast)

    axes[0].set_ylim((-ylim_f_max / 2, ylim_f_max / 2))
    axes[3].set_ylim((0, ylim_f_max))
    axes[6].set_ylim((0, ylim_f_max))

    axes[1].set_ylim((-ylim_Dslow_max / 2, ylim_Dslow_max / 2))
    axes[4].set_ylim((0, ylim_Dslow_max))
    axes[7].set_ylim((0, ylim_Dslow_max))

    axes[2].set_ylim((-ylim_Dfast_max / 2, ylim_Dfast_max / 2))
    axes[5].set_ylim((0, ylim_Dfast_max))
    axes[8].set_ylim((0, ylim_Dfast_max))

    figure.tight_layout()

    return figure, axes


def plot_quiver(figure, axes, groundtruth, predicted_mean, title, parameter_range, extent_scaling,
                x_axis_label=None, y_axis_label=None):
    quiverplot = axes.quiver(groundtruth[0, :, :],
                             groundtruth[1, :, :],
                             predicted_mean[0, :, :] - groundtruth[0, :, :],
                             predicted_mean[1, :, :] - groundtruth[1, :, :],
                             angles='xy',
                             scale_units='xy',
                             scale=1,
                             color='k',
                             headwidth=5)
    axes.set_title(title, fontsize=30)
    axes.add_patch(Rectangle((parameter_range[0, 0], parameter_range[1, 0]),
                             parameter_range[0, 1] - parameter_range[0, 0],
                             parameter_range[1, 1] - parameter_range[1, 0],
                             facecolor='none',
                             edgecolor='k',
                             linewidth=3))
    axes.set_xlim(parameter_range[0, 0] - (parameter_range[0, 1] - parameter_range[0, 0]) * extent_scaling,
                  parameter_range[0, 1] + (parameter_range[0, 1] - parameter_range[0, 0]) * extent_scaling)
    axes.set_ylim(parameter_range[1, 0] - (parameter_range[1, 1] - parameter_range[1, 0]) * extent_scaling,
                  parameter_range[1, 1] + (parameter_range[1, 1] - parameter_range[1, 0]) * extent_scaling)

    if x_axis_label:
        axes.set_xlabel(x_axis_label, fontsize=20)
    else:
        axes.set_xlabel(r'$\theta_{0}$', fontsize=20)

    if y_axis_label:
        axes.set_ylabel(y_axis_label, fontsize=20)
    else:
        axes.set_ylabel(r'$\theta_{1}$', fontsize=20)
    axes.grid(color='k', linestyle='--', linewidth=0.5)

    return figure, axes, quiverplot


def plot_2D_distribution_comparison(test_data, network_1_data, network_2_data, label_1, label_2):
    figure, axes = plt.subplots(3, 3, figsize=(25, 25))
    axes = axes.ravel()
    linewidth = 4

    test_repeats = test_data.n_repeats
    test_sampling = test_data.n_sampling

    for parameter_idx in range(3):
        # rmse plot
        rmse_1 = np.reshape(network_1_data.param_predictions_rmse, (
            test_sampling[0] * test_sampling[1] * test_sampling[2] * test_sampling[3], 4))[:, parameter_idx + 1]
        rmse_2 = np.reshape(network_2_data.param_predictions_rmse, (
            test_sampling[0] * test_sampling[1] * test_sampling[2] * test_sampling[3], 4))[:, parameter_idx + 1]
        lim_min = min(min(rmse_1), min(rmse_2))
        lim_max = max(max(rmse_1), max(rmse_2))
        axes[parameter_idx + 6].hist2d(rmse_1, rmse_2,
                                       bins=(np.linspace(lim_min, lim_max, 40), np.linspace(lim_min, lim_max, 40)),
                                       cmap=plt.cm.Blues, cmin=1)
        axes[parameter_idx + 6].plot((lim_min, lim_max),
                                     (lim_min, lim_max),
                                     color='k', alpha=1,
                                     linewidth=linewidth)
        axes[parameter_idx + 6].set_aspect('equal', 'box')

        # bias plot

        bias_1 = np.mean(np.reshape(network_1_data.param_predictions, (
            test_sampling[0] * test_sampling[1] * test_sampling[2] * test_sampling[3], test_repeats, 4))[:, :,
                         parameter_idx + 1] -
                         matlib.repmat(np.reshape(test_data.test_label, (
                             test_sampling[0] * test_sampling[1] * test_sampling[2] * test_sampling[3], 4))[:,
                                       parameter_idx + 1],
                                       test_repeats, 1).T, axis=1)
        bias_2 = np.mean(np.reshape(network_2_data.param_predictions,
                                    (
                                        test_sampling[0] * test_sampling[1] * test_sampling[2] * test_sampling[3],
                                        test_repeats,
                                        4))[
                         :, :,
                         parameter_idx + 1] -
                         matlib.repmat(
                             np.reshape(test_data.test_label,
                                        (test_sampling[0] * test_sampling[1] * test_sampling[2] * test_sampling[3], 4))[
                             :,
                             parameter_idx + 1],
                             test_repeats, 1).T, axis=1)
        lim_min = min(min(bias_1), min(bias_2))
        lim_max = max(max(bias_1), max(bias_2))
        axes[parameter_idx].hist2d(bias_1, bias_2,
                                   bins=(np.linspace(lim_min, lim_max, 40), np.linspace(lim_min, lim_max, 40)),
                                   cmap=plt.cm.Blues, cmin=1)
        axes[parameter_idx].plot((lim_min, lim_max),
                                 (lim_min, lim_max),
                                 color='k', alpha=1,
                                 linewidth=linewidth)
        axes[parameter_idx].set_aspect('equal', 'box')

        # variance plot
        std_1 = np.std(
            np.reshape(network_1_data.param_predictions,
                       (test_sampling[0] * test_sampling[1] * test_sampling[2] * test_sampling[3], test_repeats, 4))[:,
            :,
            parameter_idx + 1],
            axis=1)
        std_2 = np.std(
            np.reshape(network_2_data.param_predictions,
                       (test_sampling[0] * test_sampling[1] * test_sampling[2] * test_sampling[3], test_repeats, 4))[:,
            :,
            parameter_idx + 1],
            axis=1)
        lim_min = min(min(std_1), min(std_2))
        lim_max = max(max(std_1), max(std_2))
        axes[parameter_idx + 3].hist2d(std_1, std_2,
                                       bins=(np.linspace(lim_min, lim_max, 40), np.linspace(lim_min, lim_max, 40)),
                                       cmap=plt.cm.Blues, cmin=1)
        axes[parameter_idx + 3].plot((lim_min, lim_max),
                                     (lim_min, lim_max),
                                     color='k', alpha=1,
                                     linewidth=linewidth)
        axes[parameter_idx + 3].set_aspect('equal', 'box')

    for parameter_idx in range(3):
        axes[parameter_idx + 6].set_xlabel('RMSE: {}'.format(label_1))
        axes[parameter_idx + 6].set_ylabel('RMSE: {}'.format(label_2))

        axes[parameter_idx].set_xlabel('Bias: {}'.format(label_1))
        axes[parameter_idx].set_ylabel('Bias: {}'.format(label_2))

        axes[parameter_idx + 3].set_xlabel('Standard deviation: {}'.format(label_1))
        axes[parameter_idx + 3].set_ylabel('Standard deviation: {}'.format(label_2))

    axes[0].set_title('$f$', fontsize=30)
    axes[1].set_title('$D_{slow}$', fontsize=30)
    axes[2].set_title('$D_{fast}$', fontsize=30)

    # axes[3].xaxis.set_major_locator(ticker.MultipleLocator(0.003))
    # axes[3].yaxis.set_major_locator(ticker.MultipleLocator(0.003))

    figure.tight_layout()

    return figure, axes


def plot_summary_uniform_weighted(test_data, reference_network, weighted_network, label, color, linestyle):
    figure, axes = plt.subplots(3, 3, figsize=(35, 25))
    axes = axes.ravel()
    linewidth = 4

    axes[0].plot(
        test_data.test_label[0, :, 0, 0, 1],
        (np.mean(reference_network.param_predictions_mean, axis=(0, 2, 3))[:,
         1] - test_data.test_label[0, :, 0, 0, 1]),
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[1].plot(
        test_data.test_label[0, 0, :, 0, 2],
        (np.mean(reference_network.param_predictions_mean, axis=(0, 1, 3))[:,
         2] - test_data.test_label[0, 0, :, 0, 2]),
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[2].plot(
        test_data.test_label[0, 0, 0, :, 3],
        (np.mean(reference_network.param_predictions_mean, axis=(0, 1, 2))[:,
         3] - test_data.test_label[0, 0, 0, :, 3]),
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[3].plot(
        test_data.test_label[0, :, 0, 0, 1],
        np.mean((reference_network.param_predictions_std), axis=(0, 2, 3))[:, 1],
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[4].plot(
        test_data.test_label[0, 0, :, 0, 2],
        np.mean((reference_network.param_predictions_std), axis=(0, 1, 3))[:, 2],
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[5].plot(
        test_data.test_label[0, 0, 0, :, 3],
        np.mean((reference_network.param_predictions_std), axis=(0, 1, 2))[:, 3],
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[6].errorbar(
        test_data.test_label[0, :, 0, 0, 1],
        np.mean(reference_network.param_predictions_rmse, axis=(0, 2, 3))[:, 1],
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[7].errorbar(
        test_data.test_label[0, 0, :, 0, 2],
        np.mean(reference_network.param_predictions_rmse, axis=(0, 1, 3))[:, 2],
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[8].errorbar(
        test_data.test_label[0, 0, 0, :, 3],
        np.mean(reference_network.param_predictions_rmse, axis=(0, 1, 2))[:, 3],
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])

    axes[0].plot(
        test_data.test_label[0, :, 0, 0, 1],
        (np.mean(weighted_network[0].param_predictions_mean, axis=(0, 2, 3))[:,
         1] - test_data.test_label[0, :, 0, 0, 1]),
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])
    axes[1].plot(
        test_data.test_label[0, 0, :, 0, 2],
        (np.mean(weighted_network[1].param_predictions_mean, axis=(0, 1, 3))[:,
         2] - test_data.test_label[0, 0, :, 0, 2]),
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])
    axes[2].plot(
        test_data.test_label[0, 0, 0, :, 3],
        (np.mean(weighted_network[2].param_predictions_mean, axis=(0, 1, 2))[:,
         3] - test_data.test_label[0, 0, 0, :, 3]),
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])

    axes[3].plot(
        test_data.test_label[0, :, 0, 0, 1],
        np.mean((weighted_network[0].param_predictions_std), axis=(0, 2, 3))[:, 1],
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])
    axes[4].plot(
        test_data.test_label[0, 0, :, 0, 2],
        np.mean((weighted_network[1].param_predictions_std), axis=(0, 1, 3))[:, 2],
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])
    axes[5].plot(
        test_data.test_label[0, 0, 0, :, 3],
        np.mean((weighted_network[2].param_predictions_std), axis=(0, 1, 2))[:, 3],
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])
    axes[6].errorbar(
        test_data.test_label[0, :, 0, 0, 1],
        np.mean(weighted_network[0].param_predictions_rmse, axis=(0, 2, 3))[:, 1],
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])
    axes[7].errorbar(
        test_data.test_label[0, 0, :, 0, 2],
        np.mean(weighted_network[1].param_predictions_rmse, axis=(0, 1, 3))[:, 2],
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])
    axes[8].errorbar(
        test_data.test_label[0, 0, 0, :, 3],
        np.mean(weighted_network[2].param_predictions_rmse, axis=(0, 1, 2))[:, 3],
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])

    axes[0].set_title('$f$', fontsize=30)
    axes[1].set_title('$D_{slow}$', fontsize=30)
    axes[2].set_title('$D_{fast}$', fontsize=30)
    axes[0].legend(ncol=2)

    for axes_idx in range(3):
        axes[axes_idx].set_ylabel('Bias')
        axes[axes_idx].axhline(y=0, color='k')
        axes[axes_idx + 3].set_ylim(bottom=0)
        axes[axes_idx + 3].set_ylabel('Standard deviation')
        axes[axes_idx + 6].set_ylabel('RMSE')
        axes[axes_idx + 6].set_xlabel('Parameter value')
        axes[axes_idx * 3].set_xlim(
            (min(test_data.test_label[0, :, 0, 0, 1]), max(test_data.test_label[0, :, 0, 0, 1])))
        axes[axes_idx * 3 + 1].set_xlim(
            (min(test_data.test_label[0, 0, :, 0, 2]), max(test_data.test_label[0, 0, :, 0, 2])))
        axes[axes_idx * 3 + 2].set_xlim(
            (min(test_data.test_label[0, 0, 0, :, 3]), max(test_data.test_label[0, 0, 0, :, 3])))

    ylim_range_f = []
    ylim_range_Dslow = []
    ylim_range_Dfast = []

    for plot_idx in range(3):
        ylim_range_f.append(axes[plot_idx * 3].get_ylim()[1] - min(axes[plot_idx * 3].get_ylim()[0], 0))
        ylim_range_Dslow.append(axes[plot_idx * 3 + 1].get_ylim()[1] - min(axes[plot_idx * 3 + 1].get_ylim()[0], 0))
        ylim_range_Dfast.append(axes[plot_idx * 3 + 2].get_ylim()[1] - min(axes[plot_idx * 3 + 2].get_ylim()[0], 0))

    ylim_f_max = 1.1 * max(ylim_range_f)
    ylim_Dslow_max = 1.1 * max(ylim_range_Dslow)
    ylim_Dfast_max = 1.1 * max(ylim_range_Dfast)

    axes[0].set_ylim((-ylim_f_max / 2, ylim_f_max / 2))
    axes[3].set_ylim((0, ylim_f_max))
    axes[6].set_ylim((0, ylim_f_max))

    axes[1].set_ylim((-ylim_Dslow_max / 2, ylim_Dslow_max / 2))
    axes[4].set_ylim((0, ylim_Dslow_max))
    axes[7].set_ylim((0, ylim_Dslow_max))

    axes[2].set_ylim((-ylim_Dfast_max / 2, ylim_Dfast_max / 2))
    axes[5].set_ylim((0, ylim_Dfast_max))
    axes[8].set_ylim((0, ylim_Dfast_max))

    figure.tight_layout()

    return figure, axes


# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
#     new_cmap = colors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)))
#     return new_cmap


def plot_two_method_fixedpoint_comparison(results_1, results_2, test_data, title, parameter_range, SNR, label, color,
                                          linestyle, d_slow_idx):
    figure, axes = plt.subplots(3, 3, figsize=(35, 30))
    axes = axes.ravel()

    figure.suptitle('{}, SNR = {}\n'.format(title, SNR), fontsize=40)

    x_label = ('$f$', '$f$', '$f$')
    y_label = ('$D_{fast}$', '$D_{fast}$', '$D_{fast}$')
    title = ('$D_{{slow}}$ = {0:.2f}'.format(test_data.test_label[0, 0, d_slow_idx[0], 0, 2]),
             '',
             '$D_{{slow}}$ = {0:.2f}'.format(test_data.test_label[0, 0, d_slow_idx[1], 0, 2]))

    extent = [parameter_range[1, 0], parameter_range[1, 1], parameter_range[3, 0], parameter_range[3, 1]]

    plot_2d = [None] * 9

    linewidth = 4

    axes[1].plot(
        test_data.test_label[0, 0, :, 0, 2],
        (np.mean(results_1.param_predictions_mean, axis=(0, 1, 3))[:, 2] - test_data.test_label[0, 0, :, 0, 2]),
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[1].plot(
        test_data.test_label[0, 0, :, 0, 2],
        (np.mean(results_2.param_predictions_mean, axis=(0, 1, 3))[:, 2] - test_data.test_label[0, 0, :, 0, 2]),
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])

    axes[4].plot(
        test_data.test_label[0, 0, :, 0, 2], np.mean((results_1.param_predictions_std), axis=(0, 1, 3))[:, 2],
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[4].plot(
        test_data.test_label[0, 0, :, 0, 2], np.mean((results_2.param_predictions_std), axis=(0, 1, 3))[:, 2],
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])

    axes[7].errorbar(
        test_data.test_label[0, 0, :, 0, 2],
        np.mean(results_1.param_predictions_rmse, axis=(0, 1, 3))[:, 2],
        label=label[0], color=color[0], alpha=1, linewidth=linewidth, linestyle=linestyle[0])
    axes[7].errorbar(
        test_data.test_label[0, 0, :, 0, 2],
        np.mean(results_2.param_predictions_rmse, axis=(0, 1, 3))[:, 2],
        label=label[1], color=color[1], alpha=1, linewidth=linewidth, linestyle=linestyle[1])

    axes[1].axhline(y=0, color='k')

    axes[1].axvline(x=test_data.test_label[0, 0, d_slow_idx[0], 0, 2], color='k')
    axes[1].axvline(x=test_data.test_label[0, 0, d_slow_idx[1], 0, 2], color='k')
    axes[4].axvline(x=test_data.test_label[0, 0, d_slow_idx[0], 0, 2], color='k')
    axes[4].axvline(x=test_data.test_label[0, 0, d_slow_idx[1], 0, 2], color='k')
    axes[7].axvline(x=test_data.test_label[0, 0, d_slow_idx[0], 0, 2], color='k')
    axes[7].axvline(x=test_data.test_label[0, 0, d_slow_idx[1], 0, 2], color='k')

    axes[4].set_ylim(bottom=0)
    axes[1].set_xlim((min(test_data.test_label[0, 0, :, 0, 2]), max(test_data.test_label[0, 0, :, 0, 2])))
    axes[4].set_xlim((min(test_data.test_label[0, 0, :, 0, 2]), max(test_data.test_label[0, 0, :, 0, 2])))
    axes[7].set_xlim((min(test_data.test_label[0, 0, :, 0, 2]), max(test_data.test_label[0, 0, :, 0, 2])))

    ylim_range_Dslow = []

    for plot_idx in range(3):
        ylim_range_Dslow.append(axes[plot_idx * 3 + 1].get_ylim()[1] - axes[plot_idx * 3 + 1].get_ylim()[0])

    ylim_Dslow_max = 1.1 * max(ylim_range_Dslow)

    axes[1].set_ylim((-ylim_Dslow_max / 2, ylim_Dslow_max / 2))
    axes[4].set_ylim((0, ylim_Dslow_max))
    axes[7].set_ylim((0, ylim_Dslow_max))
    axes[1].legend(ncol=2)

    for count, idx in enumerate(d_slow_idx):
        data = (results_1.param_predictions_mean - test_data.test_label)[0, :, idx, :, 2].T - \
               (results_2.param_predictions_mean - test_data.test_label)[0, :, idx, :, 2].T
        data_max = np.max((abs(np.max(data)), abs(np.min(data))))
        plot_2d[count * 2] = axes[count * 2].imshow(
            data,
            extent=extent, cmap=matplotlib.cm.bwr_r,
            aspect='auto', origin='lower', vmin=-data_max, vmax=data_max)
        axes[count * 2].set_xlabel(x_label[count * 2], fontsize=20)
        axes[count * 2].set_ylabel(y_label[count * 2], fontsize=20)
        axes[count * 2].set_title('Bias difference, {}'.format(title[count * 2]), fontsize=30)
        figure.colorbar(plot_2d[count * 2], ax=axes[count * 2])

        data = results_1.param_predictions_std[0, :, idx, :, 2].T - \
               results_2.param_predictions_std[0, :, idx, :, 2].T
        data_max = np.max((abs(np.max(data)), abs(np.min(data))))
        plot_2d[count * 2 + 3] = axes[count * 2 + 3].imshow(
            data,
            extent=extent, cmap=matplotlib.cm.bwr_r,
            aspect='auto', origin='lower', vmin=-data_max, vmax=data_max)
        axes[count * 2 + 3].set_xlabel(x_label[count * 2], fontsize=20)
        axes[count * 2 + 3].set_ylabel(y_label[count * 2], fontsize=20)
        axes[count * 2 + 3].set_title('Standard deviation difference, {}'.format(title[count * 2]), fontsize=30)
        figure.colorbar(plot_2d[count * 2 + 3], ax=axes[count * 2 + 3])

        data = results_1.param_predictions_rmse[0, :, idx, :, 2].T - results_2.param_predictions_rmse[0, :, idx, :, 2].T
        data_max = np.max((abs(np.max(data)), abs(np.min(data))))
        plot_2d[count * 2 + 6] = axes[count * 2 + 6].imshow(
            data,
            extent=extent, cmap=matplotlib.cm.bwr_r,
            aspect='auto', origin='lower', vmin=-data_max, vmax=data_max)
        axes[count * 2 + 6].set_xlabel(x_label[count * 2], fontsize=20)
        axes[count * 2 + 6].set_ylabel(y_label[count * 2], fontsize=20)
        axes[count * 2 + 6].set_title('RMSE difference, {}'.format(title[count * 2]), fontsize=30)
        figure.colorbar(plot_2d[count * 2 + 6], ax=axes[count * 2 + 6])

    figure.tight_layout()

    return figure, axes


def plot_summary_clinical(test_label, network_data, label, color, linestyle, n_networks, parameter_range, filter=False):
    figure, axes = plt.subplots(4, 3, figsize=(35, 35))
    axes = axes.ravel()
    linewidth = 4
    n_bins = 11

    if filter:
        filter_idx = (test_label[:, 1] >= 0.1) & (test_label[:, 1] <= 0.5) & \
                     (test_label[:, 2] >= 0.4) & (test_label[:, 2] <= 3.0) & \
                     (test_label[:, 3] >= 10.0) & (test_label[:, 3] <= 150.0)
    else:
        filter_idx = test_label[:, 1] > 0

    for net_idx in range(n_networks):

        param_mean = np.zeros((n_bins, 4))
        param_std = np.zeros((n_bins, 4))
        param_rmse = np.zeros((n_bins, 4))

        for parameter in range(3):
            stepsize = (parameter_range[parameter + 1, 1] - parameter_range[parameter + 1, 0]) / (n_bins - 1)
            label_edge = np.linspace(parameter_range[parameter + 1, 0] - stepsize / 2,
                                     parameter_range[parameter + 1, 1] + stepsize / 2, n_bins + 1)
            label_centre = np.linspace(parameter_range[parameter + 1, 0], parameter_range[parameter + 1, 1], n_bins)
            bin_idx = np.digitize(test_label[filter_idx, parameter + 1], label_edge)
            bin_count = np.zeros((n_bins))

            for bin in range(n_bins):
                bin_count[bin] = sum(bin_idx == bin + 1)
                param_mean[bin, parameter + 1] = np.mean(
                    network_data[net_idx].param_predictions_mean[filter_idx, parameter + 1][bin_idx == bin + 1])
                param_std[bin, parameter + 1] = np.mean(
                    network_data[net_idx].param_predictions_std[filter_idx, parameter + 1][bin_idx == bin + 1])
                param_rmse[bin, parameter + 1] = np.mean(
                    network_data[net_idx].param_predictions_rmse[filter_idx, parameter + 1][bin_idx == bin + 1])

            axes[parameter].plot(
                label_centre, param_mean[:, parameter + 1] - label_centre,
                label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
            axes[parameter + 3].plot(
                label_centre, param_std[:, parameter + 1],
                label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
            axes[parameter + 6].plot(
                label_centre, param_rmse[:, parameter + 1],
                label=label[net_idx], color=color[net_idx], alpha=1, linewidth=linewidth, linestyle=linestyle[net_idx])
            axes[parameter + 9].bar(label_centre, bin_count, width=stepsize, color='gray')

    axes[0].set_title('$f$', fontsize=30)
    axes[1].set_title('$D_{slow}$', fontsize=30)
    axes[2].set_title('$D_{fast}$', fontsize=30)
    axes[1].legend(ncol=2)

    for parameter in range(3):
        axes[parameter].set_ylabel('Bias')
        axes[parameter].axhline(y=0, color='k')
        axes[parameter].set_xlim(parameter_range[parameter + 1, 0], parameter_range[parameter + 1, 1])

        axes[parameter + 3].set_ylabel('Standard deviation')
        axes[parameter + 3].set_xlim(parameter_range[parameter + 1, 0], parameter_range[parameter + 1, 1])

        axes[parameter + 6].set_ylabel('RMSE')
        axes[parameter + 6].set_xlabel('Parameter value')
        axes[parameter + 6].set_xlim(parameter_range[parameter + 1, 0], parameter_range[parameter + 1, 1])

        axes[parameter + 9].set_xlim(parameter_range[parameter + 1, 0], parameter_range[parameter + 1, 1])
        axes[parameter + 9].set_ylabel('Bin count')

    axes[0].set_ylim((-0.08, 0.08))
    axes[1].set_ylim((-0.4, 0.4))
    axes[2].set_ylim((-60, 65))
    axes[3].set_ylim((0, 0.16))
    axes[4].set_ylim((0, 0.8))
    axes[5].set_ylim((0, 130))
    axes[6].set_ylim((0, 0.16))
    axes[7].set_ylim((0, 0.8))
    axes[8].set_ylim((0, 130))

    figure.tight_layout()

    return figure, axes


def plot_parameter_maps_clinical(z_idx, groundtruth_flat, mask, volumes, labels, parameter_idx):
    parameter_labels = ['S0', '$f$', '$D_{slow}$', '$D_{fast}$']
    vmax = [0.5, 4.0, 300]

    cmap_truncated = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=matplotlib.cm.bwr.name, a=0.5, b=1.0),
        matplotlib.cm.bwr(np.linspace(0.5, 1.0, 1000)))

    groundtruth_repmat = np.transpose(np.reshape(
        matlib.repmat(np.reshape(groundtruth_flat, (groundtruth_flat.shape[0] * groundtruth_flat.shape[1], 1)), 1, 16),
        (groundtruth_flat.shape[0], groundtruth_flat.shape[1], 16)), (0, 2, 1))

    figure, axes = plt.subplots(3, len(volumes) + 1, figsize=((len(volumes) + 1) * 10, 30))
    axes = axes.ravel()

    for parameter_idx in range(3):
        groundtruth_volume = zerofill_masked_map(groundtruth_repmat[:, 0, :], mask)
        plot_handle = axes[(len(volumes) + 1) * parameter_idx].imshow(
            groundtruth_volume[:, :, z_idx, parameter_idx + 1].T,
            aspect='equal',
            origin='lower',
            cmap=cmap_truncated,
            vmin=0,
            vmax=vmax[parameter_idx])
        axes[(len(volumes) + 1) * parameter_idx].set_xticks([])
        axes[(len(volumes) + 1) * parameter_idx].set_yticks([])
        axes[0].set_title('Super-sampled ''groundtruth''\n', fontsize=30)
        axes[(len(volumes) + 1) * parameter_idx].set_ylabel(parameter_labels[parameter_idx + 1], fontsize=30)
        plt.colorbar(plot_handle, ax=axes[(len(volumes) + 1) * parameter_idx])

    for method_idx, estimates, in enumerate(volumes):

        estimates_volume = zerofill_masked_map(estimates[:, 0, :], mask)

        for parameter_idx in range(3):
            plot_handle = axes[1 + method_idx + (len(volumes) + 1) * parameter_idx].imshow(
                estimates_volume[:, :, z_idx, parameter_idx + 1].T,
                aspect='equal',
                origin='lower',
                cmap=cmap_truncated,
                vmin=0,
                vmax=vmax[parameter_idx])
            axes[1 + method_idx + (len(volumes) + 1) * parameter_idx].set_xticks([])
            axes[1 + method_idx + (len(volumes) + 1) * parameter_idx].set_yticks([])
            axes[1 + method_idx].set_title(labels[method_idx], fontsize=30)
            axes[1 + method_idx + (len(volumes) + 1) * parameter_idx].set_ylabel(parameter_labels[parameter_idx + 1],
                                                                                 fontsize=30)
            plt.colorbar(plot_handle, ax=axes[1 + method_idx + (len(volumes) + 1) * parameter_idx])

    return figure, axes


def zerofill_masked_map(map_nonzeros_flat, mask):
    mask_flat = np.reshape(mask, (mask.shape[0] * mask.shape[1] * mask.shape[2]))
    map_zerofill_flat = np.zeros((mask.shape[0] * mask.shape[1] * mask.shape[2], 4))
    map_zerofill_flat[mask_flat != 0, :] = map_nonzeros_flat
    map_zerofill = np.reshape(map_zerofill_flat, (mask.shape[0], mask.shape[1], mask.shape[2], 4))

    return map_zerofill
