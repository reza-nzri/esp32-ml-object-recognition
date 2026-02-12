import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from utils.data_processing import DataProcessor
from utils.model_architecture import build_model, HighPerfTuner, get_augmented_batch
from visualizations import PerformanceDashboard

BASE_DATA_PATH = os.path.join("data", "raw", "objects")
SHAPE_DIRS = {"circle": "circle", "hexagon": "hexagon", "oval": "oval", "square": "square", "triangle": "triangle"}

def run_training():
    processor = DataProcessor()
    X, y = processor.load_from_folders(BASE_DATA_PATH, SHAPE_DIRS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    tuner_dir = os.path.join(curr_dir, "..", "tuning_results")
    log_dir = os.path.join(curr_dir, "..", "src/logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tuner = HighPerfTuner(
        oracle=kt.oracles.BayesianOptimizationOracle(
            objective=kt.Objective("val_accuracy", "max"),
            max_trials=50
        ),
        hypermodel=build_model,
        directory=tuner_dir,
        project_name='ultrasonic_shape_v2'
    )

    tuner.search(X_train, y_train, epochs=50, batch_size=32)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ]
    train_dataset = tf.data.Dataset.from_generator(
        lambda: get_augmented_batch(X_train, y_train, batch_size=256),
        output_signature=(
            tf.TensorSpec(shape=(256, 163, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256,), dtype=tf.int32)
        )
    )

    history = best_model.fit(
        train_dataset,
        steps_per_epoch=len(X_train) // 32 * 10,
        epochs=500,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    class_names = sorted(SHAPE_DIRS.keys())
    dashboard = PerformanceDashboard(tuner, best_model, X_test, y_test, class_names)

    dashboard.plot_history(history)

    dashboard.plot_confusion_matrix()

    if not os.path.exists("models"):
        os.makedirs("models")
    best_model.save("models/object_recognition_model.keras")

    class_names = sorted(SHAPE_DIRS.keys())
    dashboard = PerformanceDashboard(tuner, best_model, X_test, y_test, class_names)
    dashboard.plot_confusion_matrix()


if __name__ == "__main__":
    run_training()