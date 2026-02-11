import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from matplotlib.widgets import Slider


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
        cm = confusion_matrix(self.y_test, y_pred, normalize="true")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix: Predicted vs. Actual", fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()


def plot_interactive_scans(files, labels):
    all_data_list = []
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file, label in zip(files, labels):
        df = pd.read_csv(file, header=None)
        df.columns = ['scan_index', 'step', 'angle_deg', 'distance_cm'] + list(df.columns[4:])
        df['angle_deg'] = pd.to_numeric(df['angle_deg'], errors='coerce')
        df['distance_cm'] = pd.to_numeric(df['distance_cm'], errors='coerce')
        df['scan_index'] = pd.to_numeric(df['scan_index'], errors='coerce')
        df['angle_rad'] = np.deg2rad(df['angle_deg'])
        df['Object'] = label
        all_data_list.append(df)

    combined_df = pd.concat(all_data_list, ignore_index=True)
    y_limit = combined_df['distance_cm'].max() + 5

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(bottom=0.25)

    lines = []
    for i, (df, label) in enumerate(zip(all_data_list, labels)):
        ax = axes[i]
        initial_data = df[df['scan_index'] == 0]
        line, = ax.plot(initial_data['angle_rad'], initial_data['distance_cm'], color='#2D5A27', lw=2)
        lines.append(line)
        ax.set_ylim(0, y_limit)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(label, pad=20, fontweight='bold')

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Scan ID', 0, 7, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        for i, df in enumerate(all_data_list):
            new_data = df[df['scan_index'] == idx]
            lines[i].set_data(new_data['angle_rad'], new_data['distance_cm'])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    plot_interactive_scans(
        files=["data/raw/objects/circle/circle_1cm_25_8.csv",
               "data/raw/objects/hexagon/hexagon_plastic_25_8_15ms.csv",
               "data/raw/objects/square/square_wood_25_8.csv"],
        labels=["Circle", "Hexagon", "Square"]
    )