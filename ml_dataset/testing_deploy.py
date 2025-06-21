from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# memuat model yang telah disimpan
joblib_model = joblib.load('gbr_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # mengambil data dari request JSON
    # melakukan prediksi (harus dalam bentuk 2D array)
    prediction = joblib_model.predict(data)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)

