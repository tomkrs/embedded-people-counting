import matplotlib.pyplot as plt
import pandas as pd

file_path = 'code/visualization/stadtwald/stadtwald_trackEval/pedestrian_detailed.csv'
# Load the dataset
df = pd.read_csv(file_path)

# Drop row with seq = 'COMBINED'
df = df[df['seq'] != 'COMBINED']

hota_columns = df.columns[4:24]
assa_columns = df.columns[44:64]
deta_columns = df.columns[24:44]

# Drop all columns except for the ones containing the metrics and the sequence
df = df[['seq'] + list(hota_columns) + list(assa_columns) + list(deta_columns)]


df['x'] = df['seq'].apply(lambda x: x.split('_')[0])
df['y'] = df['seq'].apply(lambda x: x.split('_')[1])
df['z'] = df['seq'].apply(lambda x: x.split('_')[2])

# Filter rows containing 'x1' in the "seq" column
df_x1 = df[df['x'] == 'x1']
df_x2 = df[df['x'] == 'x2']
df_x3 = df[df['x'] == 'x3']

# Filter rows containing 'x1' in the "seq" column
df_y1 = df[df['y'] == 'y1']
df_y2 = df[df['y'] == 'y2']
df_y3 = df[df['y'] == 'y3']

# Filter rows containing 'x1' in the "seq" column
df_z1 = df[df['z'] == 'z1']
df_z2 = df[df['z'] == 'z2']
df_z3 = df[df['z'] == 'z3']

# Mean of HOTA for each group

x1_hota_mean = df_x1[hota_columns].mean(axis=1).mean()
x2_hota_mean = df_x2[hota_columns].mean(axis=1).mean()
x3_hota_mean = df_x3[hota_columns].mean(axis=1).mean()

y1_hota_mean = df_y1[hota_columns].mean(axis=1).mean()
y2_hota_mean = df_y2[hota_columns].mean(axis=1).mean()
y3_hota_mean = df_y3[hota_columns].mean(axis=1).mean()

z1_hota_mean = df_z1[hota_columns].mean(axis=1).mean()
z2_hota_mean = df_z2[hota_columns].mean(axis=1).mean()
z3_hota_mean = df_z3[hota_columns].mean(axis=1).mean()

hota_means = [[x1_hota_mean, x2_hota_mean, x3_hota_mean],
              [y1_hota_mean, y2_hota_mean, y3_hota_mean],
              [z1_hota_mean, z2_hota_mean, z3_hota_mean]]

# Standard deviation of HOTA for each group

x1_hota_std = df_x1[hota_columns].mean(axis=1).std()
x2_hota_std = df_x2[hota_columns].mean(axis=1).std()
x3_hota_std = df_x3[hota_columns].mean(axis=1).std()

y1_hota_std = df_y1[hota_columns].mean(axis=1).std()
y2_hota_std = df_y2[hota_columns].mean(axis=1).std()
y3_hota_std = df_y3[hota_columns].mean(axis=1).std()

z1_hota_std = df_z1[hota_columns].mean(axis=1).std()
z2_hota_std = df_z2[hota_columns].mean(axis=1).std()
z3_hota_std = df_z3[hota_columns].mean(axis=1).std()

hota_stds = [[x1_hota_std, x2_hota_std, x3_hota_std],
             [y1_hota_std, y2_hota_std, y3_hota_std],
             [z1_hota_std, z2_hota_std, z3_hota_std]]

# Mean of AssA for each group

x1_assa_mean = df_x1[assa_columns].mean(axis=1).mean()
x2_assa_mean = df_x2[assa_columns].mean(axis=1).mean()
x3_assa_mean = df_x3[assa_columns].mean(axis=1).mean()

y1_assa_mean = df_y1[assa_columns].mean(axis=1).mean()
y2_assa_mean = df_y2[assa_columns].mean(axis=1).mean()
y3_assa_mean = df_y3[assa_columns].mean(axis=1).mean()

z1_assa_mean = df_z1[assa_columns].mean(axis=1).mean()
z2_assa_mean = df_z2[assa_columns].mean(axis=1).mean()
z3_assa_mean = df_z3[assa_columns].mean(axis=1).mean()

assa_means = [[x1_assa_mean, x2_assa_mean, x3_assa_mean],
              [y1_assa_mean, y2_assa_mean, y3_assa_mean],
              [z1_assa_mean, z2_assa_mean, z3_assa_mean]]

# Standard deviation of AssA for each group

x1_assa_std = df_x1[assa_columns].mean(axis=1).std()
x2_assa_std = df_x2[assa_columns].mean(axis=1).std()
x3_assa_std = df_x3[assa_columns].mean(axis=1).std()

y1_assa_std = df_y1[assa_columns].mean(axis=1).std()
y2_assa_std = df_y2[assa_columns].mean(axis=1).std()
y3_assa_std = df_y3[assa_columns].mean(axis=1).std()

z1_assa_std = df_z1[assa_columns].mean(axis=1).std()
z2_assa_std = df_z2[assa_columns].mean(axis=1).std()
z3_assa_std = df_z3[assa_columns].mean(axis=1).std()

assa_stds = [[x1_assa_std, x2_assa_std, x3_assa_std],
             [y1_assa_std, y2_assa_std, y3_assa_std],
                [z1_assa_std, z2_assa_std, z3_assa_std]]

# Mean of DetA for each group

x1_deta_mean = df_x1[deta_columns].mean(axis=1).mean()
x2_deta_mean = df_x2[deta_columns].mean(axis=1).mean()
x3_deta_mean = df_x3[deta_columns].mean(axis=1).mean()

y1_deta_mean = df_y1[deta_columns].mean(axis=1).mean()
y2_deta_mean = df_y2[deta_columns].mean(axis=1).mean()
y3_deta_mean = df_y3[deta_columns].mean(axis=1).mean()

z1_deta_mean = df_z1[deta_columns].mean(axis=1).mean()
z2_deta_mean = df_z2[deta_columns].mean(axis=1).mean()
z3_deta_mean = df_z3[deta_columns].mean(axis=1).mean()

deta_means = [[x1_deta_mean, x2_deta_mean, x3_deta_mean],
                [y1_deta_mean, y2_deta_mean, y3_deta_mean],
                [z1_deta_mean, z2_deta_mean, z3_deta_mean]]

# Standard deviation of DetA for each group

x1_deta_std = df_x1[deta_columns].mean(axis=1).std()
x2_deta_std = df_x2[deta_columns].mean(axis=1).std()
x3_deta_std = df_x3[deta_columns].mean(axis=1).std()

y1_deta_std = df_y1[deta_columns].mean(axis=1).std()
y2_deta_std = df_y2[deta_columns].mean(axis=1).std()
y3_deta_std = df_y3[deta_columns].mean(axis=1).std()

z1_deta_std = df_z1[deta_columns].mean(axis=1).std()
z2_deta_std = df_z2[deta_columns].mean(axis=1).std()
z3_deta_std = df_z3[deta_columns].mean(axis=1).std()

deta_stds = [[x1_deta_std, x2_deta_std, x3_deta_std],
                [y1_deta_std, y2_deta_std, y3_deta_std],
                [z1_deta_std, z2_deta_std, z3_deta_std]]


# Plot the HOTA results BOXPLOT
fig, axs = plt.subplots(1, 3)

hota_x = [df_x1[hota_columns].mean(axis=1), df_x2[hota_columns].mean(axis=1), df_x3[hota_columns].mean(axis=1)]
axs[0].boxplot(hota_x, showmeans=True, meanline=True, showfliers=False)
axs[0].grid('on', axis='y', linestyle='--', linewidth=0.2)
axs[0].set_xticks([1, 2, 3])

hota_y = [df_y1[hota_columns].mean(axis=1), df_y2[hota_columns].mean(axis=1), df_y3[hota_columns].mean(axis=1)]
axs[1].boxplot(hota_y, showmeans=True, meanline=True, showfliers=False)
axs[1].grid('on', axis='y', linestyle='--', linewidth=0.2)
axs[1].set_xticks([1, 2, 3])

hota_z = [df_z1[hota_columns].mean(axis=1), df_z2[hota_columns].mean(axis=1), df_z3[hota_columns].mean(axis=1)]
axs[2].boxplot(hota_z, showmeans=True, meanline=True, showfliers=False)
axs[2].grid('on', axis='y', linestyle='--', linewidth=0.2)
axs[2].set_xticks([1, 2, 3])

axs[0].set_ylabel('Score', fontsize=10, style='italic')

axs[0].set_xticklabels(['x1', 'x2', 'x3'], fontsize=10, style='italic')
axs[0].set_title('Grouped by x', fontsize=10, style='italic')
axs[1].set_xticklabels(['y1', 'y2', 'y3'], fontsize=10, style='italic')
axs[1].set_title('Grouped by y', fontsize=10, style='italic')
axs[2].set_xticklabels(['z1', 'z2', 'z3'], fontsize=10, style='italic')
axs[2].set_title('Grouped by z', fontsize=10, style='italic')

fig.suptitle('Stadtwald HOTA \nEfficientDet0-lite + SORT', fontsize=12, fontweight='bold')
fig.tight_layout()

fig.show()
plt.show()

# --------------------------------------------------------------------------------------------------------------
# Plot the HOTA results
fig, axs = plt.subplots(1, 3)
for fig_index, group in enumerate([hota_x, hota_y, hota_z]):
      axs[fig_index].boxplot(group, showmeans=True, meanline=True, showfliers=False)
      axs[fig_index].set_xticks([1, 2, 3])
      axs[fig_index].grid('on', axis='y', linestyle='--', linewidth=0.2)

axs[0].set_ylabel('Score', fontsize=10, style='italic')
axs[0].set_xticklabels(['x1', 'x2', 'x3'], fontsize=10, style='italic')
axs[0].set_title('Grouped by x', fontsize=10, style='italic')
axs[1].set_xticklabels(['y1', 'y2', 'y3'], fontsize=10, style='italic')
axs[1].set_title('Grouped by y', fontsize=10, style='italic')
axs[2].set_xticklabels(['z1', 'z2', 'z3'], fontsize=10, style='italic')
axs[2].set_title('Grouped by z', fontsize=10, style='italic')

fig.suptitle('Stadtwald HOTA \nEfficientDet0-lite + SORT', fontsize=12, fontweight='bold')
fig.tight_layout()

plt.savefig('figures/Stadtwald-Simulation/boxplots/stadtwald_hota_grouped_by_xyz.png')

# Save the data to a csv file with description and std as strings. The groups are the rows and the metrics are the columns
save_hota_df = pd.DataFrame(columns=['HOTA mean', 'HOTA std'], index=['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'z1', 'z2', 'z3'])
save_hota_df['HOTA median'] = [df_x1[hota_columns].mean(axis=1).median(), df_x2[hota_columns].mean(axis=1).median(), df_x3[hota_columns].mean(axis=1).median(), df_y1[hota_columns].mean(axis=1).median(), df_y2[hota_columns].mean(axis=1).median(), df_y3[hota_columns].mean(axis=1).median(), df_z1[hota_columns].mean(axis=1).median(), df_z2[hota_columns].mean(axis=1).median(), df_z3[hota_columns].mean(axis=1).median()]
save_hota_df.to_csv('figures/Stadtwald-Simulation/boxplots/data_hota.csv')

# --------------------------------------------------------------------------------------------------------------
# Plot the AssA results
assa_x = [df_x1[assa_columns].mean(axis=1), df_x2[assa_columns].mean(axis=1), df_x3[assa_columns].mean(axis=1)]
assa_y = [df_y1[assa_columns].mean(axis=1), df_y2[assa_columns].mean(axis=1), df_y3[assa_columns].mean(axis=1)]
assa_z = [df_z1[assa_columns].mean(axis=1), df_z2[assa_columns].mean(axis=1), df_z3[assa_columns].mean(axis=1)]

fig, axs = plt.subplots(1, 3)
for fig_index, group in enumerate([assa_x, assa_y, assa_z]):
      axs[fig_index].boxplot(group, showmeans=True, meanline=True, showfliers=False)
      axs[fig_index].set_xticks([1, 2, 3])
      axs[fig_index].grid('on', axis='y', linestyle='--', linewidth=0.2)

axs[0].set_ylabel('Score', fontsize=10, style='italic')
axs[0].set_xticklabels(['x1', 'x2', 'x3'], fontsize=10, style='italic')
axs[0].set_title('Grouped by x', fontsize=10, style='italic')
axs[1].set_xticklabels(['y1', 'y2', 'y3'], fontsize=10, style='italic')
axs[1].set_title('Grouped by y', fontsize=10, style='italic')
axs[2].set_xticklabels(['z1', 'z2', 'z3'], fontsize=10, style='italic')
axs[2].set_title('Grouped by z', fontsize=10, style='italic')

fig.suptitle('Stadtwald Association Accuracy \nEfficientDet0-lite + SORT', fontsize=12, fontweight='bold')
fig.tight_layout()

plt.savefig('figures/Stadtwald-Simulation/boxplots/stadtwald_assa_grouped_by_xyz.png')

# Save the data to a csv file with description and std as strings. The groups are the rows and the metrics are the columns
save_assa_df = pd.DataFrame(columns=['AssA mean', 'AssA std'], index=['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'z1', 'z2', 'z3'])
save_assa_df['AssA Median'] = [df_x1[assa_columns].mean(axis=1).median(), df_x2[assa_columns].mean(axis=1).median(), df_x3[assa_columns].mean(axis=1).median(), df_y1[assa_columns].mean(axis=1).median(), df_y2[assa_columns].mean(axis=1).median(), df_y3[assa_columns].mean(axis=1).median(), df_z1[assa_columns].mean(axis=1).median(), df_z2[assa_columns].mean(axis=1).median(), df_z3[assa_columns].mean(axis=1).median()]
save_assa_df.to_csv('figures/Stadtwald-Simulation/boxplots/data_assa.csv')

# --------------------------------------------------------------------------------------------------------------
# Plot the DetA results
deta_x = [df_x1[deta_columns].mean(axis=1), df_x2[deta_columns].mean(axis=1), df_x3[deta_columns].mean(axis=1)]
deta_y = [df_y1[deta_columns].mean(axis=1), df_y2[deta_columns].mean(axis=1), df_y3[deta_columns].mean(axis=1)]
deta_z = [df_z1[deta_columns].mean(axis=1), df_z2[deta_columns].mean(axis=1), df_z3[deta_columns].mean(axis=1)]

fig, axs = plt.subplots(1, 3)
for fig_index, group in enumerate([deta_x, deta_y, deta_z]):
      axs[fig_index].boxplot(group, showmeans=True, meanline=True, showfliers=False)
      axs[fig_index].set_xticks([1, 2, 3])
      axs[fig_index].grid('on', axis='y', linestyle='--', linewidth=0.2)

axs[0].set_ylabel('Score', fontsize=10, style='italic')
axs[0].set_xticklabels(['x1', 'x2', 'x3'], fontsize=10, style='italic')
axs[0].set_title('Grouped by x', fontsize=10, style='italic')
axs[1].set_xticklabels(['y1', 'y2', 'y3'], fontsize=10, style='italic')
axs[1].set_title('Grouped by y', fontsize=10, style='italic')
axs[2].set_xticklabels(['z1', 'z2', 'z3'], fontsize=10, style='italic')
axs[2].set_title('Grouped by z', fontsize=10, style='italic')



axs[0].set_xticklabels(['x1', 'x2', 'x3'], fontsize=10, style='italic')
axs[0].set_title('Grouped by x', fontsize=10, style='italic')
axs[1].set_xticklabels(['y1', 'y2', 'y3'], fontsize=10, style='italic')
axs[1].set_title('Grouped by y', fontsize=10, style='italic')
axs[2].set_xticklabels(['z1', 'z2', 'z3'], fontsize=10, style='italic')
axs[2].set_title('Grouped by z', fontsize=10, style='italic')

fig.suptitle('Stadtwald Detection Accuracy \nEfficientDet0-lite + SORT', fontsize=12, fontweight='bold')
fig.tight_layout()

plt.savefig('figures/Stadtwald-Simulation/boxplots/stadtwald_deta_grouped_by_xyz.png')

# Save the data to a csv file with description and std as strings. The groups are the rows and the metrics are the columns
save_deta_df = pd.DataFrame(columns=['DetA mean', 'DetA std'], index=['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'z1', 'z2', 'z3'])
save_deta_df['DetA median'] = [df_x1[deta_columns].mean(axis=1).median(), df_x2[deta_columns].mean(axis=1).median(), df_x3[deta_columns].mean(axis=1).median(), df_y1[deta_columns].mean(axis=1).median(), df_y2[deta_columns].mean(axis=1).median(), df_y3[deta_columns].mean(axis=1).median(), df_z1[deta_columns].mean(axis=1).median(), df_z2[deta_columns].mean(axis=1).median(), df_z3[deta_columns].mean(axis=1).median()]
save_deta_df.to_csv('figures/Stadtwald-Simulation/boxplots/data_deta.csv')