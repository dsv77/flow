
class EnvironmentWrapper(object):
    def __init__(self, environment):
        self.environment = environment
        self.transform_functions = ([t[0] for t in environment['transformations']])

    def encode(self, text):
        for t in self.transform_functions:
            text = t(text, self.environment)
        return text
