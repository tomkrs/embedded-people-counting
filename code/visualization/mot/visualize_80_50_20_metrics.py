import matplotlib.pyplot as plt
import pandas as pd

'''
80% = 0.8 = 4/5 -> KeepNotFrame_5
50% = 0.5 = 1/2 -> KeepOnlyFrame_2
20% = 0.2 = 1/5 -> KeepOnlyFrame_5
'''

file_path_80 = 'code/visualization/mot/mot15_effdet_hota/MOT15_keepNotFrame_5-train/EffDet0lite_SORT/pedestrian_detailed.csv'
file_path_50 = 'code/visualization/mot/mot15_effdet_hota/MOT15_keepOnlyFrame_2-train/EffDet0lite_SORT/pedestrian_detailed.csv'
file_path_20 = 'code/visualization/mot/mot15_effdet_hota/MOT15_keepOnlyFrame_5-train/EffDet0lite_SORT/pedestrian_detailed.csv'

df_80 = pd.read_csv(file_path_80)
df_50 = pd.read_csv(file_path_50)
df_20 = pd.read_csv(file_path_20)

# Drop row with seq = 'COMBINED'
df_80 = df_80[df_80['seq'] != 'COMBINED']
df_50 = df_50[df_50['seq'] != 'COMBINED']
df_20 = df_20[df_20['seq'] != 'COMBINED']

hota_columns_80 = df_80.columns[4:24]
assa_columns_80 = df_80.columns[44:64]
deta_columns_80 = df_80.columns[24:44]

hota_columns_50 = df_50.columns[4:24]
assa_columns_50 = df_50.columns[44:64]
deta_columns_50 = df_50.columns[24:44]

hota_columns_20 = df_20.columns[4:24]
assa_columns_20 = df_20.columns[44:64]
deta_columns_20 = df_20.columns[24:44]

# Calculate the mean of each metric and the standard deviation
hota_mean_80 = df_80[hota_columns_80].mean(axis=1).mean()*100
assa_mean_80 = df_80[assa_columns_80].mean(axis=1).mean()*100
deta_mean_80 = df_80[deta_columns_80].mean(axis=1).mean()*100

hota_std_80 = df_80[hota_columns_80].mean(axis=1).std()*100
assa_std_80 = df_80[assa_columns_80].mean(axis=1).std()*100
deta_std_80 = df_80[deta_columns_80].mean(axis=1).std()*100

hota_mean_50 = df_50[hota_columns_50].mean(axis=1).mean()*100
assa_mean_50 = df_50[assa_columns_50].mean(axis=1).mean()*100
deta_mean_50 = df_50[deta_columns_50].mean(axis=1).mean()*100

hota_std_50 = df_50[hota_columns_50].mean(axis=1).std()*100
assa_std_50 = df_50[assa_columns_50].mean(axis=1).std()*100
deta_std_50 = df_50[deta_columns_50].mean(axis=1).std()*100

hota_mean_20 = df_20[hota_columns_20].mean(axis=1).mean()*100
assa_mean_20 = df_20[assa_columns_20].mean(axis=1).mean()*100
deta_mean_20 = df_20[deta_columns_20].mean(axis=1).mean()*100

hota_std_20 = df_20[hota_columns_20].mean(axis=1).std()*100
assa_std_20 = df_20[assa_columns_20].mean(axis=1).std()*100
deta_std_20 = df_20[deta_columns_20].mean(axis=1).std()*100

# Plot the results
fig, ax = plt.subplots(1, 3)
fig.dpi = 600



