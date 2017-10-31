from flask import Flask
from flask import request
import json
from flow_framework import Predictor

"""
Web service model for a simple model
"""
class WebService(object):
    def __init__(self, model_dir, weights_files, response_formatter=None):
        self.model_dir = model_dir
        self.predictor = Predictor(model_dir)
        self.predictor.load_model(weights_files)
        if response_formatter is None:
            self.response_formatter = generate_response
        else:
            self.response_formatter = response_formatter

    def start(self, port=5000):
        app = Flask(__name__)

        @app.route('/query')
        def query():
            return self.response_formatter(request, self.predictor)
        app.run(debug=False, host='0.0.0.0', port=port)

def generate_response(request, predictor):
    query = request.args.get('q')
    s = 'Transformations'
    s += '<ul>'
    for item in predictor.transform_text(query):
        s += ('<li>%s: %s</li>' % item)
    s += '</ul>'
    predictions = predictor.predict(query)
    s += '<ul>'
    for item in predictions:
        s += ('<li>%s: %1.3f</li>' % item)
    s += '</ul>'
    return s




