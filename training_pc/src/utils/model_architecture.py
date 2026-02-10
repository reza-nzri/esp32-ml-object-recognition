from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.model_selection import KFold
from .data_processing import SmartAugmentor
import numpy as np

# **TODO** Look up Batches in CNN
def get_augmented_batch(X, y, batch_size=32):
    while True:
        idx = np.random.choice(len(X), batch_size)
        batch_X = X[idx].copy()
        batch_y = y[idx]
        for i in range(batch_size):
            batch_X[i] = SmartAugmentor.rotate_scan(batch_X[i])
            batch_X[i] = SmartAugmentor.add_noise(batch_X[i])
            batch_X[i] = SmartAugmentor.jitter_scale(batch_X[i])
        yield batch_X, batch_y

  #  **TODO** Look up, understand model structure
def build_model(hp):
    """Factory function for the Multi-Scale 1D-CNN architecture."""
    inputs = keras.Input(shape=(163, 1))
    path_outputs = []

    # Hyperparameter: Number of parallel feature paths
    # **TODO** look up "pipe" structure
    for i in range(hp.Int('num_paths', 2, 4)):
        kernel_size = hp.Choice(f'kernel_{i}', [3, 7, 11, 15])
        filters = hp.Int(f'filters_{i}', 16, 64, step=16)

        x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        path_outputs.append(x)
    # ** TODO** Look up structure of neural networks and activation functions (especially convolutional neural network)
    merged = layers.Concatenate()(path_outputs)
    x = layers.GlobalMaxPooling1D()(merged)  # Invariance to scan starting position

    x = layers.Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu')(x)
    x = layers.Dropout(hp.Float('dropout', 0.1, 0.4, step=0.1))(x)

    outputs = layers.Dense(5, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class HighPerfTuner(kt.Tuner):
    """Custom K-Fold Cross-Validation Tuner."""

    def run_trial(self, trial, X, y, **fit_kwargs):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_accuracies = []

        for train_idx, val_idx in kf.split(X):
            model = self.hypermodel.build(trial.hyperparameters)
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

            history = model.fit(X[train_idx], y[train_idx],
                                validation_data=(X[val_idx], y[val_idx]),
                                callbacks=callbacks, **fit_kwargs)

            val_accuracies.append(max(history.history['val_accuracy']))

        self.oracle.update_trial(trial.trial_id, {'val_accuracy': np.mean(val_accuracies)})