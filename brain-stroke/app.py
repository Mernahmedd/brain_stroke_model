from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/data', methods=['POST'])
def predict():
    try:
        # Extract and convert input values to floats
        a1 = float(request.form.get('a1'))
        a2 = float(request.form.get('a2'))
        a3 = float(request.form.get('a3'))
        a4 = float(request.form.get('a4'))
        a5 = float(request.form.get('a5'))
        a6 = float(request.form.get('a6'))
        input_query = np.array([[a1, a2, a3, a4, a5, a6]])

        result = model.predict(input_query)[0]

        return jsonify({'placement': str(result)})

    except ValueError as e:
        return jsonify({'error': 'Invalid input. Please provide valid numeric values.'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
