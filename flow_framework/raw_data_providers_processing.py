from flow_framework import AbstractFlowComponent
from copy import deepcopy
import numpy as np


class BordingJoinSummaryAndTextFlowComponent(AbstractFlowComponent):
    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                yield sample
                summary_sample = deepcopy(sample)
                summary_sample['text'] = summary_sample['summary']
                yield summary_sample
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        return environment


class PubmedFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                cuis = sample['class_']
                text = sample['text']
                for cui in cuis:
                    if 'abstract' in sample:
                        abstract = sample['abstract']
                        yield {'text':abstract, 'class_':cui}
                    yield {'text':text, 'class_':cui}
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        return environment



class MultilingualWikiPruneLanguagesFlowComponent(AbstractFlowComponent):
    def __init__(self, target_languages, required_languages=[], min_languages_pr_article=1):
        super(self.__class__, self).__init__()
        self.target_languages = target_languages
        self.min_languages_pr_article = min_languages_pr_article
        self.required_languages = set(required_languages)

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                new_texts = []
                languages = set()
                for (lang, text) in sample['texts']:
                    if lang in self.target_languages:
                        new_texts.append((lang, text))
                        languages.add(lang)
                sample['texts'] = new_texts
                if self.min_languages_pr_article <= len(new_texts) \
                        and len(set(languages).intersection(self.required_languages))==len(self.required_languages):
                    yield sample
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        return environment



