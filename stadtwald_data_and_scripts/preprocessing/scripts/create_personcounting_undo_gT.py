'''
Goal is, to calculate the number of people in each sequence, who have crossed the middle line of the frame
If a person went from left to right, the to_right_counter is increased by 1.
If a person went from right to left, the to_left_counter is increased by 1.
If the person crossed the line, and then went back, the according counter is decreased by 1.
The middle line is defined as the line that goes through the middle of the frame, from top to bottom.
The result is saved in a txt file, with the following format:
framenumber, to_right_counter, to_left_counter

The calculation is done with the centroid of each person
'''

import numpy as np
import pandas as pd
import scipy.io as sio
import os

# Matlab bounding box format: [x1, y1, w, h]

for file in os.listdir('mat_gT'):
    if file.endswith('.mat'):
        mat = sio.loadmat('mat_gT/' + file)
        sequence_txt_filename = file.replace('_conv.mat', '').replace('converted_gTruth_', '') + '.txt'
        cell_array = mat['labelDataCellArray']
        cell_array_np = np.array(cell_array)
        df = pd.DataFrame(cell_array_np, columns=['person1', 'person2'])

        to_left_count = 0
        to_right_count = 0
        prev_x = {}
        centroid_x = 0
        with open('PersonCounting_without_dynamic_count_gT/' + sequence_txt_filename, 'w') as f:
            for frame_number in range(len(df)):
                for person_id in range(len(df.columns)):
                    if not df.iloc[frame_number, person_id].size == 0:
                        x1 = df.iloc[frame_number, person_id][0][0]
                        y1 = df.iloc[frame_number, person_id][0][1]
                        w = df.iloc[frame_number, person_id][0][2]
                        h = df.iloc[frame_number, person_id][0][3]

                        centroid_x = (int(x1) + (int(w)/2))
                        #print('Frame: ' + str(frame_number + 1) + ', Person: ' + str(person_id + 1) + ', Centroid_x: ' + str(centroid_x))
                        # Check if this person_id is already known (i.e., in previous frames)
                        if person_id in prev_x:
                            # Check if the person has crossed the line
                            if prev_x[person_id] < 160 and centroid_x >= 160:
                                to_right_count += 1
                            elif prev_x[person_id] >= 160 and centroid_x < 160:
                                to_left_count += 1
                        # Update the previous x-coordinate for this person            
                        prev_x[person_id] = centroid_x
                f.write(str(frame_number + 1) + ', ' + str(to_right_count) + ', ' + str(to_left_count) + '\n')
                

                