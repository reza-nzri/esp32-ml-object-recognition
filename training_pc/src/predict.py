import numpy as np
import tensorflow as tf
import os
from utils.data_processing import DataProcessor


def predict_shape(csv_path, model_path="models/object_recognition_model.keras"):
    if not os.path.exists(model_path):
        print(f"Error: Model in {model_path} not found!")
        return

    model = tf.keras.models.load_model(model_path)

    class_names = ["circle", "oval", "triangle"]

    # 2. preprocessing data (has to be exact same as for training data)
    processor = DataProcessor()

    # label_idx here is 0, doenst matter for pure prediciton
    scans, _ = processor.process_file(csv_path, label_idx=0)

    # example data is first scan
    input_data = scans[0:1]  # Format: (1, 163, 1)

    # 3. make prediction
    predictions = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100

    # 4. print result
    print("\n" + "=" * 30)
    print(f"RESULT FOR: {os.path.basename(csv_path)}")
    print(f"Predicted form: {class_names[predicted_idx].upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 30)

    # show all probabalities
    for name, prob in zip(class_names, predictions[0]):
        print(f"{name}: {prob * 100:.1f}%")


if __name__ == "__main__":
    test_file = "data/raw/objects/circle/circle_1cm_25_8.csv"
    predict_shape(test_file)