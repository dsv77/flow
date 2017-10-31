import pandas as pd
import random
import os
import abc

from flow_framework import AbstractFlowComponent

class BaseProvider(AbstractFlowComponent):
    def execute(self, environment={}):
        self.environment = environment
        self.dir_path = environment['config']['data']['data_dir']
        # self.dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dir_path)

        self.prepare()

        self.load_data()

        assert 'num_validation_samples' in self.environment
        # assert 0 not in self.environment['token2id'].values() # 0 is used for masking
        self.environment['raw_train_samples_gen'] = self.raw_train_samples_gen
        self.environment['raw_valid_samples_gen'] = self.raw_valid_samples_gen
        self.environment['raw_test_samples_gen'] = self.raw_test_samples_gen
        return self.environment

    @abc.abstractmethod
    def raw_train_samples_gen(self):
        # this method must be a generator of training samples of the form (class_name, text)
        pass

    @abc.abstractmethod
    def raw_valid_samples_gen(self):
        # this method must be a generator of valid samples of the form (class_name, text)
        pass

    @abc.abstractmethod
    def raw_test_samples_gen(self):
        # this method must be a generator of valid samples of the form (class_name, text)
        pass


    @abc.abstractmethod
    def load_data(self):
        """
        this method must
        1) load a dict with tokens as keys and id's (integers) as values and store in environment under token2id
         2) load a dict with class_names as keys and class_index (integers) as values and store in environment under class2id
         3) load other data
        """
        pass

    @abc.abstractmethod
    def prepare(self):
        pass