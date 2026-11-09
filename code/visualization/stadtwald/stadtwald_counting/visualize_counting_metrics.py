import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

'''
x1 1 Person in eine Richtung
x2 2 Personen gleiche Richtung
x3 2 Personen entgegengesetzte Richtung

y1 Keine Richtungsänderung
y2 1 Person kehrt vor der Mitte um
y3 1 Person kehrt nach der Mitte um

z1 nah
z2 mittel
z3 fern
'''

# Read the new CSV file with combined metrics of right and left accuracy
new_file_path = 'code/visualization/stadtwald/stadtwald_counting/counting_combined_detailed.csv'
df_combined = pd.read_csv(new_file_path)

# Split the 'seq' column to separate the components of each sequence
df_combined[['x', 'y', 'z']] = df_combined['seq'].str.extract('x(\d+)_y(\d+)_z(\d+)')

# Convert 'x', 'y', and 'z' columns to numeric types
df_combined['x'] = pd.to_numeric(df_combined['x'], errors='coerce')
df_combined['y'] = pd.to_numeric(df_combined['y'], errors='coerce')
df_combined['z'] = pd.to_numeric(df_combined['z'], errors='coerce')

# Drop the "Confusion Matrix" column
df_combined.drop(columns=['Confusion Matrix'], inplace=True)
df_combined.drop(columns=['seq'], inplace=True)

# Try the 'groupby' and 'mean' operation again
try:
    mean_group_x_combined = df_combined.groupby('x').mean()
    std_error_x = df_combined.groupby('x').std() / df_combined.groupby('x').count() ** 0.5
    mean_group_y_combined = df_combined.groupby('y').mean()
    std_error_y = df_combined.groupby('y').std() / df_combined.groupby('y').count() ** 0.5
    mean_group_z_combined = df_combined.groupby('z').mean()
    std_error_z = df_combined.groupby('z').std() / df_combined.groupby('z').count() ** 0.5
    print("Grouping and mean calculation successful!")
except Exception as e:
    print(f"An error occurred: {e}")


# Plot for groups based on 'x'
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.barplot(x=mean_group_x_combined.index, y='Accuracy', data=mean_group_x_combined, yerr=std_error_x['Accuracy'], capsize=0.1)
plt.title('Mean Accuracy Grouped by x')
plt.xlabel('x')
plt.ylabel('Mean Accuracy')

# Plot for groups based on 'y'
plt.subplot(1, 3, 2)
sns.barplot(x=mean_group_y_combined.index, y='Accuracy', data=mean_group_y_combined, yerr=std_error_y['Accuracy'], capsize=0.1)
plt.title('Mean Accuracy Grouped by y')
plt.xlabel('y')
plt.ylabel('Mean Accuracy')

# Plot for groups based on 'z'
plt.subplot(1, 3, 3)
sns.barplot(x=mean_group_z_combined.index, y='Accuracy', data=mean_group_z_combined, yerr=std_error_z['Accuracy'], capsize=0.1)
plt.title('Mean Accuracy Grouped by z')
plt.xlabel('z')
plt.ylabel('Mean Accuracy')

plt.tight_layout()
plt.show()

'''
Nennenswerte Resultate:
Grouped by x: The mean accuracy seems highest for sequences with x1 and lowest for x2.
Grouped by y: Sequences with y2 have the highest mean accuracy, while y3 has the lowest.
Grouped by z: z2 sequences have the highest mean accuracy, and z3 has the lowest.
'''

# Let's visualize the other metrics (Precision, Recall, F1 Score, Mean Absolute Error, Mean Squared Error) 
# for the different groups (x, y, z) using bar plots with error bars.

# Filtering out only the numeric metrics for visualization
numeric_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean Absolute Error', 'Mean Squared Error']

# Loop through each numeric metric to create subplots
for metric in numeric_metrics:
    plt.figure(figsize=(15, 5))
    
    # Plot for groups based on 'x'
    plt.subplot(1, 3, 1)
    sns.barplot(x=mean_group_x_combined.index, y=metric, data=mean_group_x_combined, yerr=std_error_x[metric], capsize=0.1)
    plt.title(f'Mean {metric} Grouped by x')
    plt.xlabel('x')
    plt.ylabel(f'Mean {metric}')
    
    # Plot for groups based on 'y'
    plt.subplot(1, 3, 2)
    sns.barplot(x=mean_group_y_combined.index, y=metric, data=mean_group_y_combined, yerr=std_error_y[metric], capsize=0.1)
    plt.title(f'Mean {metric} Grouped by y')
    plt.xlabel('y')
    plt.ylabel(f'Mean {metric}')
    
    # Plot for groups based on 'z'
    plt.subplot(1, 3, 3)
    sns.barplot(x=mean_group_z_combined.index, y=metric, data=mean_group_z_combined, yerr=std_error_z[metric], capsize=0.1)
    plt.title(f'Mean {metric} Grouped by z')
    plt.xlabel('z')
    plt.ylabel(f'Mean {metric}')
    
    plt.tight_layout()
    plt.show()

'''
Accuracy: Provides an overall view of how well the model performs for each group.
Precision: Indicates the proportion of true positive identifications among the total positive identifications made.
Recall: Measures the proportion of actual positives that were identified correctly.
F1 Score: Provides a balanced view of Precision and Recall.
Mean Absolute Error (MAE): Represents the average absolute error between the predicted and actual values.
Mean Squared Error (MSE): Measures the average of the squares of the errors between the predicted and actual values.
'''