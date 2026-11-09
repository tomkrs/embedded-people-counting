from scipy.io import loadmat
import numpy as np
import pandas as pd

# Load MATLAB file
mat = loadmat('gT\converted_gTruth_x1_y1_z1_conv.mat')

# Accessing a cell array named 'cell_array_name'
cell_array = mat['labelDataCellArray']

# Convert to NumPy array
cell_array_np = np.array(cell_array)

# Accessing a struct array named 'struct_array_name'
# struct_array = mat['struct_array_name']

# If the struct has a field 'field_name', you can convert it to a NumPy array as follows:
# field_data = np.array([x[0][0]['field_name'][0][0] for x in struct_array])

df = pd.DataFrame(cell_array_np, columns=['person1', 'person2'])

print(df)