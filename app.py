# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify


def calculate_angle(a, b, c):
	a = (a)  # First
	b = (b)  # Mid
	c = (c)  # End

	radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
	angle = np.abs(radians * 180.0 / np.pi)

	if angle > 180.0:
		angle = 360 - angle

	return angle


def data_preb(arr):
	current_set = []
	all_sets = []
	count = 0
	for i in arr:

		R_shoulder = i[6]
		R_elbow = i[8]
		R_writs = i[10]

		angle = calculate_angle(R_shoulder, R_elbow, R_writs)

		current_set.append(np.array(i))

		if angle > 150:
			stage = "up"
		if angle < 150 and stage == 'up':
			stage = "down"
			all_sets.append(np.array(current_set))
			count = count + 1
			current_set = []
		if count >= 12:
			break

	padded_sets = tf.keras.preprocessing.sequence.pad_sequences(
		all_sets, padding="post", dtype="float32", maxlen=100
	)

	return padded_sets


def test_model(data):
	model = tf.keras.models.load_model('Model.h5')
	result = model.predict(data)

	return result


def Model_run(data):

	splits = np.array(data_preb(data))
	splits = splits.reshape(12, 100, 51)
	result = test_model(splits)
	return result



app = Flask(__name__)

@app.route('/model')


def home():

	data = request.get_json()
	X = data['Test']

	result = Model_run(X)

	response = {
		'result': 'success',
		'message': 'Data received and processed successfully',
		'data': result  # You can include the processed data or any other relevant information in the response
	}

	return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)
