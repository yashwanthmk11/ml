import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def generate_data():
    np.random.seed(42)
    x = np.random.rand(100)
    labels = np.array(["Class1" if xi <= 0.5 else "Class2" for xi in x[:50]])
    return x, labels

def knn_classification(train_x, train_labels, test_x, k):
    predictions = []
    for x_test in test_x:
        distances = np.abs(train_x - x_test)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)

def main():
    x, labels = generate_data()
    train_x, test_x = x[:50], x[50:]
    train_labels = labels
    k_values = [1, 2, 3, 4, 5, 20, 30]
    results = {}
    for k in k_values:
        predictions = knn_classification(train_x, train_labels, test_x, k)
        results[k] = predictions
    for k, preds in results.items():
        print(f"Results for k={k}: {preds}")
    plt.scatter(train_x, [1] * 50, c=["blue" if lbl == "Class1" else "red" for lbl in train_labels], label="Training Data")
    for k, preds in results.items():
        plt.scatter(test_x, [k] * 50, c=["blue" if lbl == "Class1" else "red" for lbl in preds], label=f"Test Data k={k}")
    plt.xlabel("x")
    plt.ylabel("k-values")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
