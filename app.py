import argparse

from flask import Flask, jsonify, render_template, request, send_from_directory

from name_origin.common import ARTIFACTS
from name_origin.predict import Predictor


def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8192
    predictor = Predictor()

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/health')
    def health():
        return jsonify(status='ok', model='character_lstm', labels=predictor.labels)

    @app.post('/api/predict')
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error='Send a JSON object with a name field.'), 400
        try:
            result = predictor.predict(payload.get('name'), payload.get('mode', 'auto'))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        return jsonify(result)

    @app.get('/report/<filename>')
    def report(filename):
        if filename not in ['confusion_matrix.png', 'training_curves.png', 'metrics.json']:
            return jsonify(error='Report not found.'), 404
        return send_from_directory(ARTIFACTS, filename)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    create_app().run(host='127.0.0.1', port=args.port, debug=False)
