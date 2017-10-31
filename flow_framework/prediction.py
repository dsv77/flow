import pickle
import os
import numpy as np
from flow_framework import CreateModelFlowComponent, LoadWeightsFromFileFlowComponent, FlowController



"""
Prediction model for a simple model
"""
class Predictor(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.environment, self.transformations, self.model_names \
            = pickle.load(open(os.path.join(model_dir, 'environment.pkl'), 'rb'))
        self.transform_functions = ([t[0] for t in self.transformations])
        self.dependencies = sum([t[1] for t in self.transformations], [])
        if 'class2id' in self.environment:
            self.id2class = dict([(i,c) for (c,i) in self.environment['class2id'].items()])
        else:
            self.id2class = None
        self.environment['transformations'] = self.transformations
        self.environment['train_as_supervised'] = False


    def load_model(self, weights_files):
        for (weights_file, model_name) in zip(weights_files, self.model_names):
            model_name = model_name.split('.')[-1]
            create_model = CreateModelFlowComponent(self.model_dir.replace('/', '.')+'.'+ model_name)
            load_weights = LoadWeightsFromFileFlowComponent(os.path.join(self.model_dir, weights_file))
            flow = [
                create_model,
                load_weights,
            ]
            self.environment = FlowController(flow).execute(self.environment)

    def encode(self, text):
        for t in self.transform_functions:
            text = t(text, self.environment)
        return text

    def transform_text(self, text):
        transform_path = [('original', text)]
        for t, _, name in self.transformations:
            text = t(text, self.environment)
            transform_path.append((name, text))
        return transform_path

    def predict(self, query, num_predictions=20):
        if 'model' not in self.environment:
            Exception('No model has been loaded')
        encoded_query = self.encode(query)
        predictions = self.environment['model'].predict(encoded_query)
        predictions = predictions[0, :].flatten()
        top_ids = np.argsort(-predictions)[0:num_predictions]
        top_classes = [self.id2class[i] for i in top_ids]
        probs = [predictions[i] for i in top_ids.tolist()[0:num_predictions]]
        results = [(class_, p) for (class_, p) in zip(top_classes, probs)]
        return results






