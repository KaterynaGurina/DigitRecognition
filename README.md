🧠 Handwritten Digit Recognition
This project demonstrates a basic handwritten digit classifier using scikit-learn’s built-in digits dataset and a Logistic Regression model. It includes data visualization, model training, evaluation, and saving the trained model for future use.

📁 Project Structure
handwritten_digit_recognition.py – Main script to train, evaluate, and save the digit classifier.

digit_model.pkl – Pickled trained model (generated after training).

requirements.txt – Python dependencies needed to run the project.

📷 Dataset
Uses the sklearn.datasets.load_digits() dataset.

Consists of 8×8 grayscale images of handwritten digits from 0 to 9 (total: 1,797 samples).

🔍 Features
Trains a logistic regression model on the digit dataset.

Visualizes example digits using matplotlib.

Evaluates model performance using a classification report and confusion matrix.

Saves the trained model to disk (digit_model.pkl) using pickle.

🚀 Getting Started
1. Clone the repository

git clone https://github.com/your-username/DigitRecognition.git
cd DigitRecognition

2. Create and activate a virtual environment (optional but recommended)

python -m venv .venv
.venv\Scripts\activate    # On Windows
source .venv/bin/activate  # On macOS/Linux

3. Install dependencies

pip install -r requirements.txt

4. Run the script

python handwritten_digit_recognition.py

The model will be trained, evaluated, and saved as digit_model.pkl.

📦 Dependencies
Listed in requirements.txt:

numpy

scikit-learn

matplotlib

opencv-python

tensorflow (not used in current script, but is intended for future extension)

pytesseract (not used in current script, but is intended for OCR features)

📄 Output Example
Confusion Matrix

Classification Report (precision, recall, F1-score)

Model file: digit_model.pkl

📌 Future Improvements
Add ability to recognize digits from uploaded images using OpenCV and pytesseract.

Replace load_digits() with custom MNIST dataset or real-world data.

Integrate a GUI or web interface for digit input.