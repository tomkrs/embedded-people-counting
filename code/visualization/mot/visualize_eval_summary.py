import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import pandas as pd

def get_summary_data(framerate_reduction_folder):
    file_path = 'code/visualization/mot/mot15_effdet_hota/' + framerate_reduction_folder + '/EffDet0lite_SORT/pedestrian_summary.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        summary_data = lines[1].split()
        return summary_data

def get_cross_validation_result():
    # Init the lists with 38 zeros
    path = 'code/visualization/mot/cross_val_results/'
    hota_std, detA_std, assA_std = [0] * 38, [0] * 38, [0] * 38
    hota_std[16] = pd.read_csv(path + 'cross_validation_results_80.csv').query("Metric == 'HOTA'")['Train_Std'].iloc[0]
    hota_std[19] = pd.read_csv(path + 'cross_validation_results_50.csv').query("Metric == 'HOTA'")['Train_Std'].iloc[0]
    hota_std[22] = pd.read_csv(path + 'cross_validation_results_20.csv').query("Metric == 'HOTA'")['Train_Std'].iloc[0]
    
    detA_std[16] = pd.read_csv(path + 'cross_validation_results_80.csv').query("Metric == 'DetA'")['Train_Std'].iloc[0]
    detA_std[19] = pd.read_csv(path + 'cross_validation_results_50.csv').query("Metric == 'DetA'")['Train_Std'].iloc[0]
    detA_std[22] = pd.read_csv(path + 'cross_validation_results_20.csv').query("Metric == 'DetA'")['Train_Std'].iloc[0]

    assA_std[16] = pd.read_csv(path + 'cross_validation_results_80.csv').query("Metric == 'AssA'")['Train_Std'].iloc[0]
    assA_std[19] = pd.read_csv(path + 'cross_validation_results_50.csv').query("Metric == 'AssA'")['Train_Std'].iloc[0]
    assA_std[22] = pd.read_csv(path + 'cross_validation_results_20.csv').query("Metric == 'AssA'")['Train_Std'].iloc[0]

    return hota_std, detA_std, assA_std


if __name__ == '__main__':

    '''
    Frames Overview for x-axis linearity
    MOT 15
    Venice-2:       30FPS 20s -> 600 Frames
    ADL-Runde-8:    30FPS 22s -> 654 Franes
    ADL-Runde-6:    30FPS 18s -> 525 Frames   
    ETH-Pedcross2:  14FPS 60s -> 837 Frames
    ETH-Sunnyday:   14FPS 25s -> 354 Frames
    ETH-Bahnhof:    14FPS 71s -> 1000 Frames
    PETS09-SL21:    7FPS  114s -> 795 Frames
    TUD-Campus:     25FPS  3s  -> 71 Frames
    TUD-Stadtmitte: 25FPS  7s  -> 179 Frames
    
    Gesamt:               340s -> 5015 Frames
    Durchschnittsframes per Second = 14,75

    '''

    framerate_reductions = [1, 19/20, 18/19, 17/18, 16/17, 15/16, 14/15, 13/14, 12/13, 11/12, 10/11, 9/10, 8/9, 
            7/8, 6/7, 5/6, 4/5, 3/4, 2/3, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10, 1/11, 
            1/12, 1/13, 1/14, 1/15, 1/16, 1/17, 1/18, 1/19, 1/20]


    frr_list_MOT15 = [get_summary_data('MOT15_keepAllFrames-train'),
                get_summary_data('MOT15_keepNotFrame_20-train'),
                get_summary_data('MOT15_keepNotFrame_19-train'),
                get_summary_data('MOT15_keepNotFrame_18-train'),
                get_summary_data('MOT15_keepNotFrame_17-train'),
                get_summary_data('MOT15_keepNotFrame_16-train'),
                get_summary_data('MOT15_keepNotFrame_15-train'),
                get_summary_data('MOT15_keepNotFrame_14-train'),
                get_summary_data('MOT15_keepNotFrame_13-train'),
                get_summary_data('MOT15_keepNotFrame_12-train'),
                get_summary_data('MOT15_keepNotFrame_11-train'),
                get_summary_data('MOT15_keepNotFrame_10-train'),
                get_summary_data('MOT15_keepNotFrame_9-train'),
                get_summary_data('MOT15_keepNotFrame_8-train'),
                get_summary_data('MOT15_keepNotFrame_7-train'),
                get_summary_data('MOT15_keepNotFrame_6-train'),
                get_summary_data('MOT15_keepNotFrame_5-train'),
                get_summary_data('MOT15_keepNotFrame_4-train'),
                get_summary_data('MOT15_keepNotFrame_3-train'),
                get_summary_data('MOT15_keepOnlyFrame_2-train'),
                get_summary_data('MOT15_keepOnlyFrame_3-train'),
                get_summary_data('MOT15_keepOnlyFrame_4-train'),
                get_summary_data('MOT15_keepOnlyFrame_5-train'),
                get_summary_data('MOT15_keepOnlyFrame_6-train'),
                get_summary_data('MOT15_keepOnlyFrame_7-train'),
                get_summary_data('MOT15_keepOnlyFrame_8-train'),
                get_summary_data('MOT15_keepOnlyFrame_9-train'),
                get_summary_data('MOT15_keepOnlyFrame_10-train'),
                get_summary_data('MOT15_keepOnlyFrame_11-train'),
                get_summary_data('MOT15_keepOnlyFrame_12-train'),
                get_summary_data('MOT15_keepOnlyFrame_13-train'),
                get_summary_data('MOT15_keepOnlyFrame_14-train'),
                get_summary_data('MOT15_keepOnlyFrame_15-train'),
                get_summary_data('MOT15_keepOnlyFrame_16-train'),
                get_summary_data('MOT15_keepOnlyFrame_17-train'),
                get_summary_data('MOT15_keepOnlyFrame_18-train'),
                get_summary_data('MOT15_keepOnlyFrame_19-train'),
                get_summary_data('MOT15_keepOnlyFrame_20-train')]
    
    frr_list = frr_list_MOT15

    # HOTA: 
    hota = [float(frr[0]) for frr in frr_list]
    
    detA = [float(frr[1]) for frr in frr_list]
    detRe = [float(frr[3]) for frr in frr_list]
    detPr = [float(frr[4]) for frr in frr_list]

    assA = [float(frr[2]) for frr in frr_list]
    assRe = [float(frr[5]) for frr in frr_list]
    assPr = [float(frr[6]) for frr in frr_list]
    
    locA = [float(frr[7]) for frr in frr_list]
    owta = [float(frr[8]) for frr in frr_list]
    
    hota_0 = [float(frr[9]) for frr in frr_list]
    locA_0 = [float(frr[10]) for frr in frr_list]
    hota_locA_0 = [float(frr[11]) for frr in frr_list]
    
    dets = [float(frr[12]) for frr in frr_list]
    gt_dets = [float(frr[13]) for frr in frr_list]
    dets_rel = []


    for i in range(len(dets)):
        dets_rel.append(dets[i] / gt_dets[i])
    
    ids = [float(frr[14]) for frr in frr_list]
    gt_ids = [float(frr[15]) for frr in frr_list]
    ids_rel = []
    for i in range(len(ids)):
        ids_rel.append(ids[i] / gt_ids[i])
    
    '''
    9 fold cross validation für
    Index start with 0
    80% = 0.8 = 4/5 -> Index 16
    50% = 0.5 = 1/2 -> Index 19
    20% = 0.2 = 1/5 -> Index 22
    '''

    error_bar_indices = [16, 19, 22]
    hota_std, detA_std, assA_std = get_cross_validation_result()


    plot_array = [[hota, detA, assA], [dets_rel], [dets_rel], [dets, gt_dets]]
    plot_array_labels = [['HOTA', 'DetA', 'AssA'], ['Detections (relative to groundTruth) zoomed'], ['Detections (relative to groundTruth)'], ['Detections (absolute)', 'GT_Detections (absolute)']]
    plot_array_colors = [['red', 'blue', 'green'], ['red'], ['red'], ['red', 'blue']]
    plot_array_yaxis_labels = [['Score'], ['Relation of Detections'], ['Relation of Detections'], ['Number of Detections']]
    
    for plot in zip(plot_array, plot_array_labels, plot_array_colors, plot_array_yaxis_labels):
        fig, axs = plt.subplots()
        
        # Plot the data
        for metric in plot[0]:
            # Plot the metric as line
            axs.plot(framerate_reductions, metric, color=plot[2][plot[0].index(metric)], label=plot[1][plot[0].index(metric)])

            # Plot the metric as scatter
            axs.scatter(framerate_reductions, metric, color=plot[2][plot[0].index(metric)], marker='x')

            #Add the errorbar to the 17. 20. and 23. HOTA, DetA and AssA 
            # if plot[1][plot[0].index(metric)] == 'HOTA':
            #     for seq in error_bar_indices:
            #         axs.errorbar(framerate_reductions[seq], hota[seq], yerr=hota_std[seq]*100, color='red', capsize=5, fmt='none')
            # if plot[1][plot[0].index(metric)] == 'DetA':
            #     for seq in error_bar_indices:        
            #         axs.errorbar(framerate_reductions[seq], detA[seq], yerr=detA_std[seq]*100, color='blue', capsize=5, fmt='none')
            # if plot[1][plot[0].index(metric)] == 'AssA':
            #     for seq in error_bar_indices:
            #         axs.errorbar(framerate_reductions[seq], assA[seq], yerr=assA_std[seq]*100, color='green', capsize=5, fmt='none')
            

            # Set the yaxis to 0-1 if the metric is Detections (relative to groundTruth)
            if plot[1][plot[0].index(metric)] == 'Detections (relative to groundTruth)':
                # Show the plot with the yaxis from 0 to 1
                axs.set_ylim([0, 1])

        rect80 = patches.Rectangle((0.78, 20), 0.04, 15,linewidth=2,linestyle='--',edgecolor='r', facecolor=None, fill=False)
        rect50 = patches.Rectangle((0.48, 19), 0.04, 12,linewidth=2,linestyle='--',edgecolor='r', facecolor=None, fill=False)
        rect20 = patches.Rectangle((0.18, 17.5), 0.04, 6.8,linewidth=2,linestyle='--',edgecolor='r', facecolor=None, fill=False)
        axs.add_patch(rect80)
        axs.add_patch(rect50)
        axs.add_patch(rect20)
        # Add a legend to the plot and set the x and y axis labels
        axs.set_xlabel('Relative FPS', style='italic', fontsize=12)
        axs.set_xlim([0, 1.05])
        axs.set_ylabel(plot[3][0], style='italic', fontsize=12)
        axs.grid('on', axis='y', linestyle='--', linewidth=0.2)
        axs.invert_xaxis()

        # Add a second x-axis to the plot with the average FPS
        twin_axs = axs.twiny()
        twin_axs.set_xticks(axs.get_xticks())
        xticks = twin_axs.get_xticks()
        twin_axs.set_xticklabels([round(float(x) * 14.75, 2) if float(x)<=1.0 else None for x in xticks])
        twin_axs.set_xlim(axs.get_xlim())
        twin_axs.set_xlabel('Average FPS [Frames per second]', style='italic', fontsize=12)

        # Plot a lightgrey dashed line at every x value to make it easier to see the difference
        for i in xticks[1:-1]:
            axs.axvline(x=float(i), color='lightgrey', linestyle='--')
        
        # Print a lightgrey dashed line at the last x metric value to make it easier to see the difference
        axs.axvline(x=1/20, color='lightgrey', linestyle='--')

        # If the plot is HOTA
        if plot[1][0] == 'HOTA':
            # Print a black dashed line at 1.5 FPS
            axs.axvline(x=0.101, color='black', linestyle='--', label='Prototype speed')
            # Show the value of the black dashed line above the plot
            axs.text(0.101, 1.03, '1.5', horizontalalignment='center', verticalalignment='center', transform=axs.get_xaxis_transform())
            # Show the label of the black dashed line
        axs.legend(fontsize=14)

        fig.set_size_inches(14.5, 7.5)
        title_string = ''
        for x in plot[1]:
            title_string += (x + ' ')
        title = 'TrackEval: ' + title_string + 'for MOT15'
        fig.suptitle(title, fontsize=19, fontweight='bold')

        plt.tight_layout()
        fig.savefig('figures/Framerate-Reduction/' + str(plot[1][0]) + '.png')
