
import abc
from flow_framework import AbstractTransformerFlowComponent

class BaseGenerator(AbstractTransformerFlowComponent):
    def execute(self, environment={}):
        self.environment = environment
        self.environment['encoded_train_samples_gen'] = self.encoded_train_samples_gen
        self.environment['encoded_valid_samples_gen'] = self.encoded_valid_samples_gen
        self.environment['encoded_test_samples_gen'] = self.encoded_test_samples_gen
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, self.get_dependencies(), self.__class__.__name__))
        return self.environment

    @abc.abstractmethod
    def encoded_train_samples_gen(self):
        # this method must be a generator of training samples of the form (class_name, text)
        raise NotImplemented()

    @abc.abstractmethod
    def encoded_valid_samples_gen(self):
        # this method must be a generator of valid samples of the form (class_name, text)
        raise NotImplemented()


    @abc.abstractmethod
    def encoded_test_samples_gen(self):
        # this method must be a generator of valid samples of the form (class_name, text)
        pass

