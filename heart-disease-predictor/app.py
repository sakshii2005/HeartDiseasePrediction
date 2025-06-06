from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [
        float(data['age']),
        float(data['sex']),
        float(data['cp']),
        float(data['trestbps']),
        float(data['chol']),
        float(data['fbs']),
        float(data['restecg']),
        float(data['thalach']),
        float(data['exang']),
        float(data['oldpeak']),
        float(data['slope'])
    ]

    prediction = model.predict([features])

    result = "❤️ Heart Disease Detected!" if prediction[0] == 1 else "✅ No Heart Disease."
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
