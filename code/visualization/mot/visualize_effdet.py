import matplotlib.pyplot as plt
import pandas as pd

file_path = 'code/visualization/mot/mot15_effdet_hota/MOT15_keepAllFrames-train/EffDet0lite_SORT/pedestrian_detailed.csv'

df = pd.read_csv(file_path)

df = df[df['seq'] != 'COMBINED']


hota_columns = df.columns[4:24]
assa_columns = df.columns[44:64]
deta_columns = df.columns[24:44]

hota_mean = df[hota_columns].mean(axis=1).mean()*100
assa_mean = df[assa_columns].mean(axis=1).mean()*100
deta_mean = df[deta_columns].mean(axis=1).mean()*100

hota_std = df[hota_columns].mean(axis=1).std()*100
assa_std = df[assa_columns].mean(axis=1).std()*100
deta_std = df[deta_columns].mean(axis=1).std()*100

fig, ax = plt.subplots()
fig.dpi = 600
ax.bar(0, deta_mean, yerr=deta_std ,color='tab:blue', alpha=0.8, edgecolor = 'black',  label='DetA', capsize=4, width=0.4)
ax.bar(1, hota_mean, yerr=hota_std ,color='tab:red', alpha=0.8, edgecolor = 'black',  label='HOTA', capsize=4, width=0.4)
ax.bar(2, assa_mean, yerr=assa_std ,color='tab:green', alpha=0.8, edgecolor = 'black',  label='AssA', capsize=4, width=0.4)
ax.grid('on', axis='y', linestyle='--', linewidth=0.2)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['DetA', 'HOTA', 'AssA'], fontsize=10, style='italic')

ax.set_ylabel('Score', fontsize=10, style='italic')

fig.suptitle('MOT15 Benchmark Train-Dataset results \n on EfficientDet0-lite & SORT', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/Effdet0lite/detA_hota_assA_relDets_visualization.png')

# Save the data to a csv file with description and std as strings. The modells are the rows and the metrics are the columns
df = pd.DataFrame(columns=['EfficientDet0-lite'], index=['DetA mean', 'DetA std', 'HOTA mean', 'HOTA std', 'AssA mean', 'AssA std'])
df['EfficientDet0-lite'] = [deta_mean, deta_std, hota_mean, hota_std, assa_mean, assa_std]
#df['MPNTrack'] = [float(data_mpntrack[0]), str(hota_std_mpn), float(data_mpntrack[1]), str(detA_std_mpn), float(data_mpntrack[2]), str(assA_std_mpn), float(data_mpntrack[12]), dets_rel_mpntrack]
df.to_csv('figures/Effdet0lite/data.csv')


    
    
    



    



