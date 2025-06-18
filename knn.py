import numpy as np
import csv
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """Store training data"""
        self.X_train = X_train
        self.y_train = y_train
    
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X_test):
        """Predict labels for test data"""
        predictions = []
        for test_point in X_test:
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self.euclidean_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Get the labels of k nearest neighbors
            k_labels = [label for _, label in k_nearest]
            
            # Majority vote
            prediction = Counter(k_labels).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)

def load_auto_data(path_data):
    """Load auto data from TSV file"""
    numeric_fields = {'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

def preprocess_features(data, threshold=25):
    """
    Returns processed features and binary labels (efficient/not efficient)
    """
    features = []
    labels = []
    
    # Extract unique values for one-hot encoding
    unique_cylinders = sorted(list(set([entry['cylinders'] for entry in data])))
    unique_origins = sorted(list(set([entry['origin'] for entry in data])))
    
    # Calculate statistics for standardization
    displacement_vals = [entry['displacement'] for entry in data]
    horsepower_vals = [entry['horsepower'] for entry in data]
    weight_vals = [entry['weight'] for entry in data]
    acceleration_vals = [entry['acceleration'] for entry in data]
    
    # Calculate mean and std for standardization
    disp_mean, disp_std = np.mean(displacement_vals), np.std(displacement_vals)
    hp_mean, hp_std = np.mean(horsepower_vals), np.std(horsepower_vals)
    weight_mean, weight_std = np.mean(weight_vals), np.std(weight_vals)
    accel_mean, accel_std = np.mean(acceleration_vals), np.std(acceleration_vals)
    
    for entry in data:
        feature_vector = []
        
        # Cylinders: one-hot encoding
        cylinders_onehot = [1 if entry['cylinders'] == c else 0 for c in unique_cylinders]
        feature_vector.extend(cylinders_onehot)
        
        # Displacement: standardized
        disp_std_val = (entry['displacement'] - disp_mean) / disp_std
        feature_vector.append(disp_std_val)
        
        # Horsepower: standardized
        hp_std_val = (entry['horsepower'] - hp_mean) / hp_std
        feature_vector.append(hp_std_val)
        
        # Weight: standardized
        weight_std_val = (entry['weight'] - weight_mean) / weight_std
        feature_vector.append(weight_std_val)
        
        # Acceleration: standardized
        accel_std_val = (entry['acceleration'] - accel_mean) / accel_std
        feature_vector.append(accel_std_val)
        
        # Origin: one-hot encoding
        origin_onehot = [1 if entry['origin'] == o else 0 for o in unique_origins]
        feature_vector.extend(origin_onehot)
        
        # Model year and car name: dropped (as decided in 1.b)
        
        features.append(feature_vector)
        
        # Create binary label: efficient (1) if mpg >= threshold, not efficient (0) otherwise
        label = 1 if entry['mpg'] >= threshold else 0
        labels.append(label)
    
    return np.array(features), np.array(labels)

def k_fold_split(X, y, k=10):
    """Split data into k folds for cross-validation"""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k
    
    folds = []
    for i in range(k):
        start_idx = i * fold_size
        if i == k - 1:  # Last fold gets remaining samples
            end_idx = n_samples
        else:
            end_idx = (i + 1) * fold_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        folds.append((X_train, X_test, y_train, y_test))
    
    return folds

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy and F1 score"""
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # F1 Score calculation
    # True Positives, False Positives, False Negatives
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, f1_score, tn, tp, fn, fp

def run_knn_experiment():
    """Run the complete KNN experiment with cross-validation"""
    
    print("Loading auto data...")
    try:
        auto_data = load_auto_data('auto-mpg-regression.tsv')
        print(f"Loaded {len(auto_data)} data points")
    except FileNotFoundError:
        print("Error: Could not find 'auto-mpg-regression.tsv'. Please ensure the file is in the correct location.")
        return
    
    # Preprocess features
    print("Preprocessing features...")
    X, y = preprocess_features(auto_data, threshold=25)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution - Efficient: {np.sum(y)}, Not Efficient: {len(y) - np.sum(y)}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # K values to test
    k_values = [3, 5, 7, 9, 11]
    
    # Results storage
    results = []
    
    print("\nRunning 10-fold cross-validation...")
    print("=" * 60)
    
    for k in k_values:
        print(f"\nTesting K = {k}")
        
        # Create k-fold splits
        folds = k_fold_split(X, y, k=10)
        
        accuracies = []
        f1_scores = []
        tns = []
        tps = []
        fns = []
        fps = []

        
        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
            # Train KNN classifier
            knn = KNNClassifier(k=k)
            knn.fit(X_train, y_train)
            
            # Make predictions
            y_pred = knn.predict(X_test)
            
            # Calculate metrics
            accuracy, f1_score, tn, tp, fn, fp = calculate_metrics(y_test, y_pred)
            accuracies.append(accuracy)
            f1_scores.append(f1_score)

            tns.append(tn)
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)
            
            print(f"  Fold {fold_idx + 1}: Accuracy = {accuracy:.4f}, F1 = {f1_score:.4f}")
        
        # Calculate average metrics
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        avg_tn = np.mean(tns)
        avg_tp = np.mean(tps)
        avg_fn = np.mean(fns)
        avg_fp = np.mean(fps)


        results.append({
            'k': k,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'avg_tn': avg_tn, 
            'avg_tp': avg_tp, 
            'avg_fn': avg_fn, 
            'avg_fp': avg_fp, 
        })
        
        print(f"  Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        
        print(f"  Average TN: {avg_tn:.0f}")
        print(f"  Average TP: {avg_tp:.0f}")
        print(f"  Average FN: {avg_fn:.0f}")
        print(f"  Average FP: {avg_fp:.0f}")
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS - 10-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"{'K':<5} {'Accuracy':<12} {'Std Dev':<10} {'F1 Score':<12} {'Std Dev':<10} "
          f"{'TN':<5} {'TP':<5} {'FN':<5} {'FP':<5}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['k']:<5} {result['avg_accuracy']:<12.4f} {result['std_accuracy']:<10.4f} "
              f"{result['avg_f1']:<12.4f} {result['std_f1']:<10.4f} "
              f"{result['avg_tn']:<5.0f} {result['avg_tp']:<5.0f} {result['avg_fn']:<5.0f} {result['avg_fp']:<5.0f}")
    
    # Find best performing K
    best_accuracy_k = max(results, key=lambda x: x['avg_accuracy'])
    best_f1_k = max(results, key=lambda x: x['avg_f1'])
    
    print("\n" + "=" * 80)
    print("BEST PERFORMING CONFIGURATIONS:")
    print(f"Best Accuracy: K = {best_accuracy_k['k']} (Accuracy = {best_accuracy_k['avg_accuracy']:.4f})")
    print(f"Best F1 Score: K = {best_f1_k['k']} (F1 = {best_f1_k['avg_f1']:.4f})")
    
    return results

def explain_metrics():
    print("\n" + "=" * 80)
    print("METRIC SELECTION JUSTIFICATION")
    print("=" * 80)
    print("""
Two metrics were selected for evaluation:

1. ACCURACY:
   - Measures the overall proportion of correct predictions
   - Suitable for balanced datasets where both classes are equally important
   - Easy to interpret and understand
   - Provides a general sense of model performance

2. F1 SCORE:
   - Harmonic mean of precision and recall
   - Particularly useful for imbalanced datasets
   - Balances both precision (avoiding false positives) and recall (avoiding false negatives)
   - More robust than accuracy when class distribution is skewed

RATIONALE:
As mentioned in the report, since we are measuring car efficiency (not a critical 
application like medical diagnosis), we don't need to heavily favor precision or 
recall. Using both accuracy and F1 score provides:
- Accuracy for balanced data scenarios
- F1 score for potentially imbalanced data situations
- A comprehensive evaluation that works well regardless of class distribution
- This ensures reliable evaluations across different data scenarios
    """)

if __name__ == "__main__":
    results = run_knn_experiment()
    explain_metrics()
