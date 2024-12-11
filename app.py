from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

try:
    diabetes_dataset = pd.read_csv('diabetesdata.csv')
except FileNotFoundError:
    print("File not found. Check the file path.")
    exit()

x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        input_data = np.asarray([
            data.get('pregnancies', 0),
            data.get('glucose', 0),
            data.get('bloodPressure', 0),
            data.get('skinThickness', 0),
            data.get('insulin', 0),
            data.get('bmi', 0),
            data.get('age', 0)
        ], dtype=np.float32).reshape(1, -1)

        std_data = scaler.transform(input_data)
        prediction = classifier.predict(std_data)[0]
        response = "The person is suffering from diabetes" if prediction == 1 else "The person is not suffering from diabetes"
        return jsonify({"prediction": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 400

if __name__ == "__main__":
    app.run(debug=True)
