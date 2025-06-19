# Import all necessary libraries
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2                # OpenCV for image processing
import pickle             # python built in serialization library to save/load our trained model

def train_and_save_model(model_path = "digit_model.pkl"):
    # Load the built-in handwritten digits dataset (8x8 grayscale images)
    digits = load_digits()

    # # Data shape
    # print("Data shape:", digits.data.shape)        # (1797, 64)
    # print("Image shape:", digits.images[0].shape)  # (8, 8)
    # print("Target labels:", np.unique(digits.target))  # [0–9]

    # Visualization of target data (0-9)
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(digits.images[i], cmap='pink')
        plt.title(f"Label: {digits.target[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Store data into 1D vector
    X = digits.data.astype(np.float32)

    # Store desired outputs (target data)
    y = digits.target

    # Normalize the pixel values (from 0–16 to 0–1)
    X /= 16.0

    # Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creation of logictic regression model
    model = LogisticRegression(max_iter=10000)

    # Fit model (actual training of model)
    model.fit(X_train, y_train)

    # Prediction model based on training data
    y_pred = model.predict(X_test)

    # Precision: Of all times the model predicted this digit, how many were correct?
    # Recall: Of all actual times this digit appeared, how many did the model find?
    # F1-score: A balance between precision and recall
    # Support: How many times that digit appeared in the test set
    # Accuracy = Total number of predictions/Number of correct predictions
    # Macro avg: Avg performance per digit (equal weight to all digits),
    # Weighted avg: Avg performance considering digit frequency

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Ratio between right and wrong guesses on a 10x10 matrix since our target data is 0-9
    # Rows: Actual labels (what the digit really was),
    # Columns: Predicted labels (what the model guessed)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save (pickle) the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to '{model_path}'")

if __name__ == "__main__":
    train_and_save_model()

def load_trained_model(model_path = "digit_model.pkl"):
   #load the pickeled logistick regression model from the sidk
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model