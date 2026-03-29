import pandas as pd
import numpy as np
import random

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index  
        self.threshold = threshold          
        self.left = left                    
        self.right = right                  
        self.info_gain = info_gain         
        self.value = value                  

# Gini Safsızlığı Hesaplama Formülü: 1 - sum(p_i^2)
def calculate_gini(y):
    classes = pd.Series(y).unique() 
    gini = 1.0
    total_samples = len(y)
    
    for cls in classes:
        p_i = len(y[y == cls]) / total_samples
        gini -= p_i ** 2
    return gini

def split_data(X, y, feature_index, threshold):
    left_mask = X.iloc[:, feature_index] <= threshold
    right_mask = X.iloc[:, feature_index] > threshold
    
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def find_best_split(X, y):
    best_split = {}
    max_info_gain = -1
    
    n_samples, n_features = X.shape 
    for feature_index in range(n_features):
        feature_values = X.iloc[:, feature_index]
        possible_thresholds = feature_values.unique()
        
        for threshold in possible_thresholds:
            X_left, X_right, y_left, y_right = split_data(X, y, feature_index, threshold)
            
            if len(y_left) > 0 and len(y_right) > 0:
                # Bilgi Kazancı (Information Gain) hesaplama
                parent_gini = calculate_gini(y)
                n = len(y)
                n_l, n_r = len(y_left), len(y_right)
                
                child_gini = (n_l / n) * calculate_gini(y_left) + (n_r / n) * calculate_gini(y_right)
                info_gain = parent_gini - child_gini
                
                if info_gain > max_info_gain:
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "X_left": X_left, "y_left": y_left,
                        "X_right": X_right, "y_right": y_right,
                        "info_gain": info_gain
                    }
                    max_info_gain = info_gain
                    
    return best_split

def build_tree(X, y, current_depth=0, max_depth=5):
    n_samples, n_features = X.shape
    
    if current_depth <= max_depth and len(pd.Series(y).unique()) > 1:
        best_split = find_best_split(X, y)
        
        if best_split.get("info_gain", 0) > 0:
            left_subtree = build_tree(best_split["X_left"], best_split["y_left"], current_depth + 1, max_depth)
            right_subtree = build_tree(best_split["X_right"], best_split["y_right"], current_depth + 1, max_depth)
            
            return Node(feature_index=best_split["feature_index"], 
                        threshold=best_split["threshold"], 
                        left=left_subtree, right=right_subtree, 
                        info_gain=best_split["info_gain"])

    leaf_value = pd.Series(y).value_counts().index[0]
    return Node(value=leaf_value)

def predict_single(node, row):
    if node.value is not None:
        return node.value
    
    feature_val = row.iloc[node.feature_index]
    if feature_val <= node.threshold:
        return predict_single(node.left, row)
    else:
        return predict_single(node.right, row)

def predict(tree, X):
    predictions = [predict_single(tree, X.iloc[i]) for i in range(X.shape[0])]
    return predictions

def get_k_fold_indices(data_length, k=10):
    indices = list(range(data_length))
    random.shuffle(indices) 
    
    fold_size = data_length // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != (k - 1) else data_length
        test_idx = indices[start:end]
        train_idx = [idx for idx in indices if idx not in test_idx]
        folds.append((train_idx, test_idx))
        
    return folds

def create_confusion_matrix(y_true, y_pred, classes):
    matrix = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}
    
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
        
    return matrix
def calculate_metrics_from_cm(matrix, classes):
    metrics = {}
    
    total_samples = 0
    correct_predictions = 0
    for c in classes:
        correct_predictions += matrix[c][c]
        for p in classes:
            total_samples += matrix[c][p]

    metrics['Accuracy'] = correct_predictions / total_samples if total_samples > 0 else 0
    
    for c in classes:
        TP = matrix[c][c]

        FP = sum([matrix[t][c] for t in classes if t != c])
        FN = sum([matrix[c][p] for p in classes if p != c])
    
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[c] = {
            'Precision': precision, 
            'Recall': recall, 
            'F1-Score': f1_score
        }
        
    return metrics

df = pd.read_csv('FishtrainDATA.csv')
X_data = df.drop('Species', axis=1)
y_data = df['Species'].values
classes = pd.Series(y_data).unique()

folds = get_k_fold_indices(len(df), k=10)

all_accuracies = []
all_f1_scores = {c: [] for c in classes}
all_precisions = {c: [] for c in classes}
all_recalls = {c: [] for c in classes}

for i, (train_idx, test_idx) in enumerate(folds):
    print(f"--- Fold {i+1} ---")

    X_train, y_train = X_data.iloc[train_idx], y_data[train_idx]
    X_test, y_test = X_data.iloc[test_idx], y_data[test_idx]

    tree = build_tree(X_train, y_train, max_depth=5)

    y_pred = predict(tree, X_test)

    cm = create_confusion_matrix(y_test, y_pred, classes)
    metrics = calculate_metrics_from_cm(cm, classes)
    
    all_accuracies.append(metrics['Accuracy'])
    for c in classes:
        all_f1_scores[c].append(metrics[c]['F1-Score'])
        all_precisions[c].append(metrics[c]['Precision'])
        all_recalls[c].append(metrics[c]['Recall'])


print("\n=== 10-FOLD CROSS VALIDATION ORTALAMA SONUÇLARI ===")
print(f"Genel Ortalama Accuracy: {sum(all_accuracies) / 10:.4f}")

for c in classes:
    avg_f1 = sum(all_f1_scores[c]) / 10
    avg_prec = sum(all_precisions[c]) / 10
    avg_rec = sum(all_recalls[c]) / 10
    print(f"\nSınıf: {c}")
    print(f"  Ortalama Precision: {avg_prec:.4f}")
    print(f"  Ortalama Recall:    {avg_rec:.4f}")
    print(f"  Ortalama F1-Score:  {avg_f1:.4f}")