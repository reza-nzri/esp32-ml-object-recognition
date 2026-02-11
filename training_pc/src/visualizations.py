import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


class PerformanceDashboard:
    def __init__(self, tuner, best_model, X_test, y_test, class_names):
        self.tuner = tuner
        self.model = best_model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names

    def plot_confusion_matrix(self):
        """Shows exactly where the model makes mistakes."""
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix: Predicted vs. Actual", fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
