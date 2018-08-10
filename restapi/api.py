#!flask/bin/python
from flask import Flask, jsonify, request

app = Flask(__name__)


terminals = [
	{
		'id': 1,
		'prediction': 'none'
	},
	{
		'id': 2,
		'prediction': 'none'
	},
]


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/api/predictions', methods=['GET'])
def get_tasks():
    return jsonify({'terminals': terminals})

@app.route('/api/predictions/<int:terminal_id>', methods=['GET'])
def get_prediction(terminal_id):
    terminal = [terminal for terminal in terminals if terminal['id'] == terminal_id]
    return jsonify(terminal[0])

@app.route('/api/predictions/<int:terminal_id>', methods=['PUT'])
def update_terminal(terminal_id):
    terminal = [terminal for terminal in terminals if terminal['id'] == terminal_id]
    if len(terminal) == 0:
        abort(404)
    if not request.json:
        abort(400)
    terminal[0]['prediction'] = request.json["prediction"]
    return jsonify(terminal[0])


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5200)
