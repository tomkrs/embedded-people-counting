'''
Goal is to convert the matlab groudn Truth detections to a new txt file that can be used with TrackEval to evaluate the HOTA metrics
'''

import numpy as np
import pandas as pd
import scipy.io as sio 
import os

# Matlab bounding box format: [x1, y1, w, h]
# HOTA txt file format: framenumber, id, x1, y1, x2, y2, -1, -1, -1, -1

for file in os.listdir('mat_gT'):
    if file.endswith('.mat'):
        mat = sio.loadmat('mat_gT/' + file)
        sequence_txt_filename = file.replace('_conv.mat', '').replace('converted_gTruth_', '') + '.txt'
        cell_array = mat['labelDataCellArray']
        cell_array_np = np.array(cell_array)
        df = pd.DataFrame(cell_array_np, columns=['person1', 'person2'])

        # For every column in row i, write a new line to the txt file
        # The frame number is the row number + 1
        # The id is the column number + 1
        # The x1 and y1 are the first two values in the column
        # The x2 and y2 are the first two values in the column + the width and height
        # The last 4 values are -1
        # Some frames do not have a detection for a person, so there is no bounding box for that person in that frame
        # In that case, skip this row
        with open('HOTA_gT/' + sequence_txt_filename, 'w') as f:
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    if not df.iloc[i, j].size == 0:
                        x1 = df.iloc[i, j][0][0]
                        y1 = df.iloc[i, j][0][1]
                        x2 = x1 + float(df.iloc[i, j][0][2])
                        y2 = y1 + float(df.iloc[i, j][0][3])
                        f.write(str(i + 1) + ', ' + str(j) + ', ' + str(x1) + ', ' + str(y1) + ', ' + str(x2) + ', ' + str(y2) + ', -1, -1, -1, -1\n')

