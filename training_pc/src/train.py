import os
import datetime
import tensorflow as tf
import keras_tuner as kt
from utils.data_processing import DataProcessor
from utils.model_architecture import build_model, HighPerfTuner
from visualizations import PerformanceDashboard


BASE_DATA_PATH = os.path.join("data", "raw", "objects")

SHAPE_DIRS = {
    "circle": "circle",
    "hexagon": "hexagon",
    "oval": "oval",
    "square": "square",
    "triangle": "traingle"
}


def run_training():
    processor = DataProcessor()
    X, y = processor.load_from_folders(BASE_DATA_PATH, SHAPE_DIRS)

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

    tuner.search(X, y, epochs=30, batch_size=32)


    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Beste Parameter gefunden! Lernrate: {best_hps.get('lr')}")

    best_model = tuner.hypermodel.build(best_hps)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )

    print("Starte finales Training für maximale Genauigkeit...")
    best_model.fit(
        X, y,
        epochs=50,
        validation_split=0.2,
        callbacks=[tensorboard_callback],
        verbose=1
    )

    if not os.path.exists("models"):
        os.makedirs("models")
    best_model.save("models/best_shape_model.keras")

    class_names = sorted(SHAPE_DIRS.keys())
    dashboard = PerformanceDashboard(tuner, best_model, X, y, class_names)
    dashboard.plot_confusion_matrix()
    dashboard.plot_optimization_history()
    dashboard.plot_kernel_importance()


if __name__ == "__main__":
    run_training()

    #run
    # tensorboard --logdir logs
    # in terminal to view tensorboard for more visualization