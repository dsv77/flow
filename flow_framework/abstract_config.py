import abc

class AbstractConfig(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def create_model(self, environment={}):
        pass
