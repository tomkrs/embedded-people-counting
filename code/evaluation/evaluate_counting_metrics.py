'''
Metrics to evaluate
- Genauigkeit (Accuracy): Der Anteil der korrekt gezählten Personen, die die Mitte in einer bestimmten Richtung überquert haben.
- Präzision (Precision): Wie viele der gezählten Überquerungen waren tatsächlich korrekt?
- Sensitivität (Recall): Wie viele der tatsächlichen Überquerungen wurden vom Modell erfasst?
- F1-Score: Ein Mittelwert aus Präzision und Sensitivität, um ein ausgewogenes Maß der Modellleistung zu bekommen.
- Mean Absolute Error (MAE): Durchschnittlicher absoluter Fehler zwischen den gezählten Überquerungen und der Ground Truth.
- Mean Squared Error (MSE): Durchschnittlicher quadratischer Fehler; gibt mehr Gewicht auf größere Fehler.
- Korrelationskoeffizient: Misst den Grad, in dem die vorhergesagten und tatsächlichen Zählungen korrelieren.
- Confusion Matrix: Um False Positives und False Negatives zu verstehen.

'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, confusion_matrix
import numpy as np
import os
import pandas as pd

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data[:, 1], data[:, 2]  # to_right_count, to_left_count

def calculate_metrics(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)  # Avoid division by zero (if no true positives
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse
        }

def evaluate_metrics_seperate_counter():
    det_path = 'code/evaluation/personcounting/effdet0_lite_sort'
    gt_path = 'code/pipeline/personcounting_immutable_gT'
    metrics_list = []


    for file in os.listdir(det_path):
        pred_file = det_path + '/' + file
        gt_file = gt_path + '/' + file
        sequence_name = file.replace('.txt', '')
        
        
        if os.path.exists(pred_file) and os.path.exists(gt_file):
            pred_to_right, pred_to_left = load_data(pred_file)
            gt_to_right, gt_to_left = load_data(gt_file)
            
            # Assuming that 'to_right_count' and 'to_left_count' can be treated separately
            metrics_to_right = calculate_metrics(pred_to_right, gt_to_right)
            metrics_to_left = calculate_metrics(pred_to_left, gt_to_left)

            # Combine metrics for both directions
            
            combined_metrics = {f"Right_{key}": value for key, value in metrics_to_right.items()}
            combined_metrics.update({f"Left_{key}": value for key, value in metrics_to_left.items()})
            
            combined_metrics['seq'] = f'{sequence_name}'

            metrics_list.append(combined_metrics)
        else:
            print(f"Files for sequence {sequence_name} are missing.")
    
    df = pd.DataFrame(metrics_list)
    df.to_csv('code/evaluation/counting_seperated_detailed.csv', index=False)

def evaluate_metrics_combined_counter():
    det_path = 'code/evaluation/personcounting/effdet0_lite_sort'
    gt_path = 'code/pipeline/personcounting_immutable_gT'
    metrics_list = []


    for file in os.listdir(det_path):
        pred_file = det_path + '/' + file
        gt_file = gt_path + '/' + file
        sequence_name = file.replace('.txt', '')
        
        if os.path.exists(pred_file) and os.path.exists(gt_file):
            pred_to_right, pred_to_left = load_data(pred_file)
            gt_to_right, gt_to_left = load_data(gt_file)
            
            pred_combined = np.concatenate((pred_to_right, pred_to_left))
            gt_combined = np.concatenate((gt_to_right, gt_to_left))
            
            combined_metrics = calculate_metrics(pred_combined, gt_combined)
            
            combined_metrics['seq'] = f'{sequence_name}'
            metrics_list.append(combined_metrics)
        else:
            print(f"Files for sequence {sequence_name} are missing.")
    
    df = pd.DataFrame(metrics_list)
    df.to_csv('code/evaluation/counting_combined_detailed.csv', index=False)

if __name__ == '__main__':
    evaluate_metrics_seperate_counter()
    evaluate_metrics_combined_counter()