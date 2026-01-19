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

    def plot_optimization_history(self):
        """Visualizes how the Bayesian Optimizer improved over time."""
        trials = self.tuner.oracle.get_best_trials(num_trials=50)
        # Sort by trial ID to see chronological progress
        trials_sorted = sorted(self.tuner.oracle.trials.values(), key=lambda x: int(x.trial_id))
        scores = [t.score for t in trials_sorted if t.score is not None]

        plt.figure(figsize=(10, 5))
        plt.plot(scores, 'o-', color='teal', label='Trial Accuracy')
        plt.plot(np.maximum.accumulate(scores), 'r--', label='Best Accuracy So Far')
        plt.title("Bayesian Optimization Progress", fontsize=14)
        plt.xlabel("Trial Number")
        plt.ylabel("Mean 5-Fold Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

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

    def plot_kernel_importance(self):
        """Analyzes which kernel sizes the Tuner preferred in top trials."""
        top_trials = self.tuner.oracle.get_best_trials(num_trials=10)
        kernels = []
        for t in top_trials:
            hp = t.hyperparameters.values
            # Collect all kernels used in this trial
            for key in hp:
                if 'kernel' in key:
                    kernels.append(hp[key])

        plt.figure(figsize=(8, 5))
        sns.countplot(x=kernels, palette="viridis")
        plt.title("Kernel Size Preference (Top 10 Architectures)", fontsize=14)
        plt.xlabel("Kernel Size (Receptive Field)")
        plt.ylabel("Frequency of Use")
        plt.show()