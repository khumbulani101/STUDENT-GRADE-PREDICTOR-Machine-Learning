from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
with open('grade_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    studytime = float(request.form['studytime'])
    absences = float(request.form['absences'])
    G1 = float(request.form['G1'])

    # Prepare the data for prediction
    features = np.array([[studytime, absences, G1]])
    prediction = model.predict(features)[0]

    return render_template('index.html', 
                           prediction_text=f'Predicted Final Grade: {prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
