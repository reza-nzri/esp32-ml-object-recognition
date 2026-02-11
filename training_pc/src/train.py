import os
import datetime
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split  # Neu hinzugefügt
from utils.data_processing import DataProcessor
from utils.model_architecture import build_model, HighPerfTuner
from visualizations import PerformanceDashboard


BASE_DATA_PATH = os.path.join("data", "raw", "objects")

SHAPE_DIRS = {
    "circle": "circle",
    "oval":"oval",
    "triangle": "triangle"
}


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
    tuner_dir = os.path.abspath(os.path.join(curr_dir, "..", "tuning_results"))

    log_base_dir = os.path.abspath(os.path.join(curr_dir, "..", "src/logs", "fit"))
    log_dir = os.path.join(log_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tuner = HighPerfTuner(
        oracle=kt.oracles.BayesianOptimizationOracle(
            objective=kt.Objective("val_accuracy", "max"),
            max_trials=50
        ),
        hypermodel=build_model,
        directory=tuner_dir,
        project_name='ultrasonic_shape_v2'
    )

    tuner.search(X_train, y_train, epochs=100, batch_size=32)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Beste Parameter gefunden! Lernrate: {best_hps.get('lr')}")

    best_model = tuner.hypermodel.build(best_hps)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )

    best_model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_callback],
        verbose=1
    )

    if not os.path.exists("models"):
        os.makedirs("models")
    best_model.save("models/object_recognition_model.keras")

    class_names = sorted(SHAPE_DIRS.keys())
    dashboard = PerformanceDashboard(tuner, best_model, X_test, y_test, class_names)
    dashboard.plot_confusion_matrix()


if __name__ == "__main__":
    run_training()