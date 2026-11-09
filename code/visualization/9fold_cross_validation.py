'''
9 fold cross validation für
80% = 0.8 = 4/5 -> KeepNotFrame_5
50% = 0.5 = 1/2 -> KeepOnlyFrame_2
20% = 0.2 = 1/5 -> KeepOnlyFrame_5
'''


from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def perform_cross_validation(file_path, number):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Calculate the mean for HOTA, AssA, and DetA for each sequence
    hota_columns = df.columns[4:24]
    assa_columns = df.columns[44:64]
    deta_columns = df.columns[24:44]
    df['HOTA_mean'] = df[hota_columns].mean(axis=1)
    df['AssA_mean'] = df[assa_columns].mean(axis=1)
    df['DetA_mean'] = df[deta_columns].mean(axis=1)
    
    # Initialize 9-Fold cross-validation
    kf = KFold(n_splits=9, shuffle=True, random_state=42)
    
    # Initialize lists to store metrics
    hota_mean_train, hota_std_train, hota_mean_test, hota_std_test = [], [], [], []
    assa_mean_train, assa_std_train, assa_mean_test, assa_std_test = [], [], [], []
    deta_mean_train, deta_std_train, deta_mean_test, deta_std_test = [], [], [], []
    
    # Perform 9-Fold cross-validation
    for train_index, test_index in kf.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        
        # HOTA
        hota_mean_train.append(train['HOTA_mean'].mean())
        hota_std_train.append(train['HOTA_mean'].std())
        hota_mean_test.append(test['HOTA_mean'].mean())
        hota_std_test.append(test['HOTA_mean'].std())
        
        # AssA
        assa_mean_train.append(train['AssA_mean'].mean())
        assa_std_train.append(train['AssA_mean'].std())
        assa_mean_test.append(test['AssA_mean'].mean())
        assa_std_test.append(test['AssA_mean'].std())
        
        # DetA
        deta_mean_train.append(train['DetA_mean'].mean())
        deta_std_train.append(train['DetA_mean'].std())
        deta_mean_test.append(test['DetA_mean'].mean())
        deta_std_test.append(test['DetA_mean'].std())
    
    # Prepare the results
    results = {
        'Metric': ['HOTA', 'AssA', 'DetA'],
        'Train_Mean': [np.mean(hota_mean_train), np.mean(assa_mean_train), np.mean(deta_mean_train)],
        'Train_Std': [np.mean(hota_std_train), np.mean(assa_std_train), np.mean(deta_std_train)],
        'Test_Mean': [np.mean(hota_mean_test), np.mean(assa_mean_test), np.mean(deta_mean_test)],
        'Test_Std': [np.mean(hota_std_test), np.mean(assa_std_test), np.mean(deta_std_test)]
    }
    
    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    
    results_file_path = 'cross_validation_results_' + number + '.csv'
    results_df.to_csv(results_file_path, index=False)
    
    return results_file_path


# Load the dataset

# file_path_80 = 'code\scripts\mot15_results\MOT15_keepNotFrame_5-train\EffDet0lite_SORT\pedestrian_detailed.csv'
# file_path_50 = 'code\scripts\mot15_results\MOT15_keepOnlyFrame_2-train\EffDet0lite_SORT\pedestrian_detailed.csv'
# file_path_20 = 'code\scripts\mot15_results\MOT15_keepOnlyFrame_5-train\EffDet0lite_SORT\pedestrian_detailed.csv'

# perform_cross_validation(file_path_20, '20')

# file_path_effdet = 'code/visualization/mot/mot15_effdet_hota/MOT15_keepAllFrames-train/EffDet0lite_SORT/pedestrian_detailed.csv'
# file_path_mpntrack = 'code/visualization/mot/mot15_mpn_hota/pedestrian_detailed.csv'

# perform_cross_validation(file_path_effdet, 'effdet')
# perform_cross_validation(file_path_mpntrack, 'mpntrack')

# file_path_stadtwald = 'code/visualization/stadtwald/stadtwald_trackEval/pedestrian_detailed.csv'
# perform_cross_validation_grouped(file_path_stadtwald, 'stadtwald_grouped')
# perform_cross_validation(file_path_stadtwald, 'stadtwald_all')
# perform_cross_validation(file_path_stadtwald, 'stadtwald_groupedByX')
# perform_cross_validation(file_path_stadtwald, 'stadtwald_groupedByY')
# perform_cross_validation(file_path_stadtwald, 'stadtwald_groupedByZ')