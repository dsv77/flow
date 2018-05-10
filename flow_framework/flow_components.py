from copy import deepcopy
from gensim.models.phrases import Phrases
import shutil
import abc
import configparser
import os
import pickle
import operator
from stemming.porter2 import stem
import numpy as np
from importlib import import_module
import random
import urllib
import string
from multiprocessing import Process
import time
import hashlib

class AbstractFlowComponent(object):
    @abc.abstractmethod
    def execute(self, environment={}):
        raise NotImplementedError()

"""
An AbstractTransformerFlowComponent is a flow component that modifies the iterators in some way. E.g. a stemming component
expects a tokenized stream and modifies the data stream by performing stemming.
"""
class AbstractTransformerFlowComponent(AbstractFlowComponent):
    def __init__(self):
        pass

    @abc.abstractmethod
    def transform(self, text, env=None):
        raise NotImplementedError()

    """
    get_dependencies return the environment variables that are used by the transformer
    """
    def get_dependencies(self):
        return []


class CompileModelFlowComponent(AbstractFlowComponent):
    def __init__(self, loss, metrics,optimizer,loss_weights=[1]):
        super(self.__class__, self).__init__()
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.loss_weights = loss_weights

    def execute(self, environment={}):
        # Compile the model so it is ready for training
        print ('Compiling model...')
        environment['model'].compile(optimizer=self.optimizer,
                             loss=self.loss,
                                     loss_weights=self.loss_weights,
                             metrics=self.metrics)
        return environment


class ConfigLoaderFlowComponent(AbstractFlowComponent):
    # config file can be the name of a config file, or a dict
    def __init__(self, config_file):
        super(self.__class__, self).__init__()
        self.config_file = config_file


    def execute(self, environment={}):
        if 'config' in environment:
            config = environment['config']
        else:
            config = {}
        if not os.path.isfile(self.config_file):
            msg = 'Config file not found!: %s' % self.config_file
            print (msg)
            raise FileNotFoundError(msg)
        self.config = configparser.ConfigParser()
        if isinstance(self.config_file, dict):
            for section in self.config_file:
                for option in self.config_file[section]:
                    self.config.set(section, option, self.config_file[section][option])
        else:
            self.config.read(self.config_file)
        for section in self.config.sections():
            if not section in config:
                config[section] = {}
            for option in self.config.options(section):
                config[section][option] = self.config.get(section, option)
        environment['config'] = config
        return environment



class CreateModelFlowComponent(AbstractFlowComponent):
    def __init__(self, config_module, clear_models=False):
        super(self.__class__, self).__init__()
        self.config_module = config_module
        self.clear_models = clear_models

    def execute(self, environment={}):
        print ('Creating model ' + self.config_module)
        environment.setdefault('model_names', [])
        environment.setdefault('models',[])
        if self.clear_models:
            import gc
            for m in environment['models']:
                del m
            gc.collect()
            environment['model_names'] = []
            environment['models'] = []
        config = import_module('{}'.format(self.config_module)).Config()
        environment = config.create_model(environment)
        environment['models'].append(environment['model'])
        environment['model_names'].append(self.config_module)
        print("Number of parameters in model (trainable+non-trainable): %i" % environment['model'].count_params())
        return environment



class LoadBestWeightsFlowComponent(AbstractFlowComponent):
    def __init__(self, weights_dir, order='ascending'):
        super(self.__class__, self).__init__()
        self.weights_dir = weights_dir
        self.order = order

    def execute(self, environment={}):
        if os.path.exists(self.weights_dir):
            checkpoint_files = os.listdir(self.weights_dir)
            if checkpoint_files:
                if self.order == 'ascending':
                    latest_checkpoint = sorted(os.listdir(self.weights_dir))[0]
                else:
                    latest_checkpoint = sorted(os.listdir(self.weights_dir))[-1]
                print('Resuming training from latest checkpoint: {}'.format(latest_checkpoint))
                environment['model'].load_weights(self.weights_dir + '/' + latest_checkpoint)
        return environment


class LogModelTestPerformanceFlowComponent(AbstractFlowComponent):
    def __init__(self, max_n, log_dir, log_file, batch_size=32):
        super(self.__class__, self).__init__()
        self.max_n = max_n
        self.log_dir = log_dir
        self.log_file = log_file
        self.batch_size = batch_size

    def execute(self, environment={}):
        encoded_text, classes = next(environment['encoded_test_samples_gen']())
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        print('Logging performance using %i samples' % classes.shape[0])
        predicted_classes = environment['model'].predict(encoded_text, batch_size=self.batch_size)
        predicted_classes = np.argsort(-predicted_classes)
        classes = np.repeat(classes,predicted_classes.shape[1], axis=1)
        res = np.equal(classes, predicted_classes).astype('int32')
        recall_at_n = ['%1.4f' % np.mean(np.sum(res[:,0:n], axis=1)) for n in range(1, self.max_n+1)]
        open(os.path.join(self.log_dir, self.log_file), 'w').write('\t'.join(recall_at_n))
        return environment

class LogModelTestAccurracyByClassFlowComponent(AbstractFlowComponent):
    def __init__(self, max_n, log_dir, log_file, batch_size=32, minimum_num_samples_in_class=10, separator=','):
        super(self.__class__, self).__init__()
        self.max_n = max_n
        self.log_dir = log_dir
        self.log_file = log_file
        self.batch_size = batch_size
        self.minimum_num_samples_in_class = minimum_num_samples_in_class
        self.separator = separator

    def execute(self, environment={}):
        encoded_text, classes = next(environment['encoded_test_samples_gen']())
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        print('Logging performance using %i samples' % classes.shape[0])
        predicted_classes = environment['model'].predict(encoded_text, batch_size=self.batch_size)
        predicted_classes = np.argsort(-predicted_classes)[:,0]
        id2class = dict([(i,c) for (c,i) in environment['class2id'].items()])
        accuracies = {}
        for i in range(0, classes.shape[0]):
            name = id2class[int(classes[i])]
            accuracies.setdefault(name, [])
            accuracies[name].append(1 if classes[i] == predicted_classes[i] else 0)

        accuracies = sorted(accuracies.items(), key=lambda x:-len(x[1]))
        accuracies = [(c,np.mean(l),len(l)) for c, l in accuracies if len(l)>self.minimum_num_samples_in_class]
        open(os.path.join(self.log_dir, self.log_file), 'w').writelines(['%s%s%1.4f%s%i\n' %(c,self.separator, p,self.separator, n) for c, p,n in accuracies])
        return environment


class LogModelConfusionFlowComponent(AbstractFlowComponent):
    def __init__(self, max_n, log_dir, log_file, batch_size=32, minimum_num_samples_in_class=10, separator=','):
        super(self.__class__, self).__init__()
        self.max_n = max_n
        self.log_dir = log_dir
        self.log_file = log_file
        self.batch_size = batch_size
        self.minimum_num_samples_in_class = minimum_num_samples_in_class
        self.separator = separator

    def execute(self, environment={}):
        encoded_text, classes = next(environment['encoded_test_samples_gen']())
        class_counts = {}
        for c in classes:
            c = int(c)
            class_counts.setdefault(c, 0)
            class_counts[c] += 1
        idx = [i for (i, c) in list(class_counts.items()) if c >= self.minimum_num_samples_in_class]

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        print('Logging performance using %i samples' % classes.shape[0])
        predicted_classes = environment['model'].predict(encoded_text, batch_size=self.batch_size)
        predicted_classes = np.argsort(-predicted_classes)[:,0]
        id2class = dict([(i,c) for (c,i) in environment['class2id'].items()])
        oldid2newid = dict([(j,i) for (i,j) in enumerate(idx)])

        confussion = np.zeros((len(oldid2newid), len(oldid2newid)))
        for i in range(0, classes.shape[0]):
            if not int(classes[i]) in oldid2newid or not predicted_classes[i] in oldid2newid:
                continue
            confussion[oldid2newid[int(classes[i])], oldid2newid[predicted_classes[i]]] += 1
        labels = [id2class[i] for i in range(0, len(id2class)) if i in idx]

        # header = '\t' + ('\t'.join([id2class[i] for i in range(0, len(id2class)) if i in idx]))
        # header += '\n'
        # rows = [(id2class[i] + '\t') + '\t'.join([str(c) for c in confussion[classes[i],:].flatten().tolist()]) for i in range(0, len(id2class)) if i in idx]
        header = self.separator + (self.separator.join(labels))
        header += '\n'
        rows = [(labels[i] + self.separator) + self.separator.join([str(c) for c in confussion[i, :].flatten().tolist()]) for i in range(0, len(labels))]

        open(os.path.join(self.log_dir, self.log_file), 'w').write(header + '\n'.join(rows))
        return environment


class VotingClassifier():
    def __init__(self, estimators, voting='hard', weights=None):
        super(self.__class__, self).__init__()
        if weights is None:
            weights = [1]*len(estimators)
        assert isinstance(weights, list)
        assert len(estimators) == len(weights)
        self.weights = weights
        self.weights = (np.asarray(self.weights)/np.sum(self.weights)).tolist()
        self.voting = voting
        self.estimators_ = estimators

    def predict(self, X, batch_size=32):
        predictions = np.sum(np.asarray([w*estimator.predict(X, batch_size) for w, estimator in zip(self.weights, self.estimators_)]),axis=0)
        return predictions



class CombineModelsFlowComponent(AbstractFlowComponent):
    def __init__(self, voting='hard', weights=None):
        super(self.__class__, self).__init__()
        self.weights = weights
        self.voting = voting

    def execute(self, environment={}):
        estimators = environment['models']
        environment['model'] = VotingClassifier(estimators, voting=self.voting, weights=self.weights)
        return environment


"""
Transform the tokenized raw streams to n-grams
"""
class NGramFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, ngram_outputfile_name, cache_name, phrases_args={}, fields=['text']):
        super(self.__class__, self).__init__()
        self.ngram_outputfile_name = ngram_outputfile_name
        self.cache_name = cache_name
        self.phrases_args = phrases_args
        self.fields = fields
        assert isinstance(self.fields,list)

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator


    def transform(self, text, env=None):
        if not isinstance(text, list):
            print ("WARNING!!! THE INPUT TO THE STEMMING COMPONENT IS ASSUMED TO BE A LIST OF WORDS. "
                   "Input seems to be a list of chars")
        return list(self.bigram[text])


    def execute(self, environment={}):
        np.random.seed(2)
        ngram_dir = os.path.join(environment['config']['cache']['cache_dir'],self.cache_name, 'ngrams')
        if not os.path.isdir(ngram_dir):
            os.makedirs(ngram_dir)
        print ('Using n-grams')
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        ngram_file_full_path = os.path.join(ngram_dir, self.ngram_outputfile_name)
        if os.path.isfile(ngram_file_full_path):
            self.bigram = Phrases.load(ngram_file_full_path)
        else:
            texts = sum([[row[field] for i, row in enumerate(environment['raw_train_samples_gen']())] for field in self.fields],[])
            self.bigram = Phrases(texts,**self.phrases_args)
            self.bigram.save(ngram_file_full_path)


        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        environment.setdefault('ngram_dicts',[])
        environment['ngram_dicts'].append(self.bigram.vocab)
        return environment

"""
NgramVocabAdjustment removes the ngrams from gensim ngram detector, that are not in the dict environment[key]
"""
class NgramVocabAdjustment(AbstractFlowComponent):
    def __init__(self, key):
        super(self.__class__, self).__init__()
        self.key = key

    def execute(self, environment={}):
        target_dict = environment[self.key]
        if 'ngram_dicts' not in environment:
            return environment
        for ngram_dict in environment['ngram_dicts']:
            ngrams = list(ngram_dict.keys())
            for w in ngrams:
                if w.decode("utf-8") not in target_dict:
                    ngram_dict.__delitem__(w)
        return environment


class CopyFilesFlowComponent(AbstractFlowComponent):
    def __init__(self, source_files, dest_files):
        super(self.__class__, self).__init__()
        if not isinstance(source_files, list):
            self.source_files = [source_files]
        if not isinstance(dest_files, list):
            self.dest_files = [dest_files]

    def execute(self, environment={}):
        for src, dest in zip(self.source_files, self.dest_files):
            shutil.copy(src, dest)
        return environment


class CopyBestWeightsFlowComponent(AbstractFlowComponent):
    def __init__(self, weights_dir, out_dir, out_file, order='ascending'):
        super(self.__class__, self).__init__()
        self.weights_dir = weights_dir
        self.out_dir = out_dir
        self.out_file = out_file
        self.order = order

    def execute(self, environment={}):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        if os.path.exists(self.weights_dir):
            checkpoint_files = os.listdir(self.weights_dir)
            if checkpoint_files:
                if self.order == 'ascending':
                    best_checkpoint = sorted(os.listdir(self.weights_dir))[0]
                else:
                    best_checkpoint = sorted(os.listdir(self.weights_dir))[-1]
                shutil.copy(os.path.join(self.weights_dir, best_checkpoint), os.path.join(self.out_dir,self.out_file))
        return environment




class CountWordsFlowComponent(AbstractFlowComponent):
    def __init__(self, cache_name, fields=['text']):
        super(self.__class__, self).__init__()
        self.cache_name = cache_name
        self.fields = fields
        if isinstance(self.fields,list):
            def iter_fields(sample):
                for field in fields:
                    yield sample[field]
            self.iterfields = iter_fields
        else:
            self.iterfields = fields

    def execute(self, environment={}):
        cache_dir = os.path.join(environment['config']['cache']['cache_dir'], self.cache_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        counts_file = os.path.join(cache_dir, 'token2count.pkl')
        if os.path.isfile(counts_file):
            environment['token2count'] = pickle.load(open(counts_file, 'rb'))
        else:
            token2count = {}
            for i, sample in enumerate(environment['raw_train_samples_gen']()):
                if i < 100:
                    for t in self.iterfields(sample):
                        if not isinstance(t, list):
                            print("WARNING!!! THE INPUT TO THE COUNT COMPONENT IS ASSUMED TO BE A LIST OF WORDS. "
                                  "Input seems to be a list of chars?")
                for field in self.iterfields(sample):
                    for t in field:
                        token2count.setdefault(t,0)
                        token2count[t] += 1
            environment['token2count'] = token2count
            pickle.dump(token2count, open(counts_file, 'wb'))
        return environment


class LoadClass2samplesFlowComponent(AbstractFlowComponent):
    def __init__(self, cache_name=None, encode_function=None, verbose=False):
        super(self.__class__, self).__init__()
        self.cache_name = cache_name
        self.verbose = verbose
        if encode_function is None:
            self.encode_function = lambda x,e : x
        else:
            self.encode_function = encode_function

    def execute(self, environment={}):
        if self.cache_name is not None:
            cache_dir = environment['config']['cache']['cache_dir']
            cache_dir = os.path.join(cache_dir, self.cache_name)
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            file_name = os.path.join(cache_dir, 'class2samples.pkl')
        else:
            file_name = ''

        if os.path.isfile(file_name) and self.cache_name is not None:
            print ('Loading class2samples...')
            class2samples = pickle.load(open(file_name, mode='rb'))
            print ('Done loading class2samples')
        else:
            print ('Creating class2samples...')
            class2samples = {}
            for i, sample in enumerate(environment['raw_train_samples_gen']()):
                text = self.encode_function(sample['text'], environment)
                (class_name, text) = sample['class_'], text
                class2samples.setdefault(class_name,[])
                class2samples[class_name].append(text)
                if self.verbose:
                    if i % 1000 == 0:
                        print (i)
            if self.cache_name is not None:
                pickle.dump(class2samples, open(file_name, 'wb'))
            print ('Done creating class2samples')
        # category_counts = sorted(class2samples.items(), key=lambda x: -len(x[1]))
        # for (c,samples) in category_counts:
        #     print((c, len(samples)))
        environment['class2samples'] = class2samples
        return environment


class LoadClass2samplesOnlineHashFlowComponent(AbstractFlowComponent):
    def __init__(self, max_tokens, class_names):
        super(self.__class__, self).__init__()
        self.max_tokens = max_tokens
        self.class_names = class_names

    def execute(self, environment={}):
        class2samples = {}
        def gen():
            while True:
                for i, sample in enumerate(environment['raw_train_samples_gen']()):
                    (class_name, text) = sample['class_'], sample['text']
                    yield (class_name, text)
        max_tokens = self.max_tokens
        sample_gen = gen()

        class Class2Samples():
            def __getitem__(self, item):
                while True:
                    (class_name, text) = next(sample_gen)
                    if class_name == item:
                        break
                return [(class_name, text)]
        environment['class2samples'] = Class2Samples()

        class Token2Id():
            def __getitem__(self, item):
                return (item.__hash__() % max_tokens)

            def __len__(self):
                return max_tokens

            def values(self):
                return [max_tokens-1]
        environment['token2id'] = Token2Id()

        environment['class2samples'] = class2samples
        return environment



class LoadWeightsFromFileFlowComponent(AbstractFlowComponent):
    def __init__(self, weights_file):
        super(self.__class__, self).__init__()
        self.weights_file = weights_file

    def execute(self, environment={}):
        if not os.path.exists(self.weights_file):
            raise FileNotFoundError('The weights %s do not exist' % self.weights_file)
        environment['model'].load_weights(self.weights_file)
        return environment

class AddModelToEnsembleFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()

    def execute(self, environment={}):
        if 'ensemble_models' not in environment:
            environment['ensemble_models'] = []
        environment['ensemble_models'].append(environment['model'])
        if 'ensemble_transformations' not in environment:
            environment['ensemble_transformations'] = []

        transformations = ([t[0] for t in environment['transformations']])
        def encode(text):
            for t in transformations:
                text = t(text, environment)
            return text

        environment['ensemble_transformations'].append(encode)
        return environment


class MaxWordsFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()


    def execute(self, environment={}):
        np.random.seed(2)
        max_words = int(environment['config']['tokens']['max_tokens'])
        if max_words < 0:
            return
        token_dict = environment['token2id']

        print ('Using only the most frequent %i words' % max_words)
        token2count = environment['token2count']
        if max_words < len(token_dict):
            sorted_words = sorted(token2count.items(), key=operator.itemgetter(1), reverse=True)
            sorted_words = dict([(s, c) for s, c in sorted_words if s in token_dict][0:max_words])
            sorted_words = [w for w, _ in sorted(token_dict.items(), key=lambda x: x[1]) if w in sorted_words]
            environment['token2id'] = dict(zip(sorted_words, range(0, max_words)))
        return environment


"""
A tokenize component expects a text to be a string. Output are the tokenized words.
"""
class WordTokenizeFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.fields = fields
        assert isinstance(self.fields,list)

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator


    def transform(self, text, env=None):
        import nltk
        if not isinstance(text, str):
            print ("WARNING!!! THE INPUT TO THE tokenize COMPONENT IS ASSUMED TO BE A string.")
        return nltk.word_tokenize(text)


    def execute(self, environment={}):
        # self.all_samples = []
        # for sample in  self.generator_generator(environment['raw_train_samples_gen'])():
        #     self.all_samples.append(sample)
        # def raw_train_samples_gen():
        #     for sample in self.all_samples:
        #         yield sample
        # environment['raw_train_samples_gen'] = raw_train_samples_gen

        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment

"""
A tokenize component expects a text to be a string. Output are the tokenized words.
"""
class Token2IdTransformerFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.fields = fields
        assert isinstance(self.fields,list)
        self.id2tokens = {}

    def generator_generator(self, it, env):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field], env)
                yield sample
        return new_generator


    def transform(self, text, env=None):
        if not isinstance(text, list):
            print ("WARNING!!! THE INPUT TO THE HashTransformerFlowComponent COMPONENT IS ASSUMED TO BE A list.")
        # return [env['token2id'][t] for t in text if t in env['token2id'] ]
        for t in text:
            idx = env['token2id'][t]
            if hasattr(self, 'id2tokens'):
                self.id2tokens.setdefault(idx, [])
                if t not in self.id2tokens[idx]:
                    self.id2tokens[idx].append(t)
        return [env['token2id'][t] for t in text]


    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'], environment)
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'], environment)
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'], environment)
        environment['id2tokens'] = self.id2tokens
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, ['token2id', 'id2tokens'], self.__class__.__name__))
        return environment

"""
Removes all tokens not in token2id
"""
class PruneNonDictTokensTransformerFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.fields = fields
        assert isinstance(self.fields,list)

    def generator_generator(self, it, env):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field], env)
                yield sample
        return new_generator


    def transform(self, text, env=None):
        if not isinstance(text, list):
            print ("WARNING!!! THE INPUT TO THE HashTransformerFlowComponent COMPONENT IS ASSUMED TO BE A list.")
        # return [env['token2id'][t] for t in text if t in env['token2id'] ]
        return [t for t in text if t in env['token2id']]


    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'], environment)
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'], environment)
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'], environment)
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, ['token2id'], self.__class__.__name__))
        return environment



"""
Removes all non-bigrams tokens 
"""
class PruneNonBigramTokensFlowComponent(AbstractFlowComponent):
    def __init__(self, threshold=10.0, min_count=5):
        super(self.__class__, self).__init__()
        self.threshold = threshold
        self.min_count = min_count


    def execute(self, environment={}):
        prune_list = {}
        token2count = environment['token2count']
        token2id = environment['token2id']
        N = len(token2id)
        for token in token2id:
            if '_' in token:
                w1, w2 = token.split('_')
                if w1 not in token2count or w2 not in token2count or token not in token2count:
                    continue
                if N*(token2count[token]-self.min_count)/(token2count[w1]*token2count[w2]) < self.threshold:
                    prune_list[token] = None
        np.random.seed(2)

        print ('Pruning %i bigrams' % len(prune_list))
        # print(prune_list)
        words = [s for s, c in token2id.items() if s not in prune_list]
        words = sorted(words)
        environment['token2id'] = dict(zip(words, range(0, len(words))))
        return environment


"""
A tokenize component expects a text to be a string. Output are the tokenized words.
"""
class CharTokenizeFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.fields = fields
        assert isinstance(self.fields,list)

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator


    def transform(self, text, env=None):
        if not isinstance(text, str):
            print ("WARNING!!! THE INPUT TO THE tokenize COMPONENT IS ASSUMED TO BE A string.")
        return [c for c in text]


    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment


class LimitSamplesFlowComponent(AbstractFlowComponent):
    def __init__(self, max_samples_dict={'train':5000, 'valid':5000, 'test':5000}):
        super(self.__class__, self).__init__()
        self.max_samples_dict = max_samples_dict

    def raw_train_samples_gen(self, generator, max_num):
        def new_generator():
                for count, sample in enumerate(generator()):
                    if count > max_num:
                        break
                    yield sample
        return new_generator

    def execute(self, environment={}):
        for gen_type in self.max_samples_dict:
            environment['raw_%s_samples_gen'%gen_type] = self.raw_train_samples_gen(environment['raw_%s_samples_gen'%gen_type], self.max_samples_dict[gen_type])
        return environment


class LimitSampleTextFlowComponent(AbstractFlowComponent):
    def __init__(self, max_size=500, fields=['text'],
                 target_generators=['raw_train_samples_gen', 'raw_valid_samples_gen', 'raw_test_samples_gen']):
        super(self.__class__, self).__init__()
        self.max_size = max_size
        self.target_generators = target_generators
        self.fields = fields

    def raw_train_samples_gen(self, generator):
        def new_generator():
                for count, sample in enumerate(generator()):
                    for field in self.fields:
                        sample[field] = sample[field][0:self.max_size]
                    yield sample
        return new_generator

    def execute(self, environment={}):
        for gen_type in self.target_generators:
            environment[gen_type] = self.raw_train_samples_gen(environment[gen_type])
        return environment


"""
MergeTrainSamplesGeneratorsFlowComponent merges the train samples generators from several flows. One of the flows is
the main flow, meaning that only the state of that environment is used after the merge (in order to solve conflicts
from e.g. config files (they could for example both have a data_dir entry)
"""
class MergeTrainSamplesGeneratorsFlowComponent(AbstractFlowComponent):
    def __init__(self, main_flow, secondary_flows=[]):
        super(self.__class__, self).__init__()
        self.main_flow = main_flow
        self.secondary_flows = secondary_flows

    def raw_train_samples_gen(self, generators):
        def new_generator():
            for generator in generators:
                for sample in generator():
                    yield sample
        return new_generator

    def execute(self, environment={}):
        train_generators = []
        main_environment = self.main_flow.execute(environment)
        train_generators.append(main_environment['raw_train_samples_gen'])
        for component in self.secondary_flows:
            environment = component.execute({})
            train_generators.append(environment['raw_train_samples_gen'])
        main_environment['raw_train_samples_gen'] = self.raw_train_samples_gen(train_generators)
        return main_environment

"""
CacheTrainGeneratorGeneratorsFlowComponent makes a cached copy of the train the raw_train_samples_gen. The purpose of this
 is to avoid having to e.g. tokenize more than once.
"""
class CacheTrainGeneratorGeneratorsFlowComponent(AbstractFlowComponent):
    def __init__(self, cache_name=None, sample_file_name='cached_samples.pkl'):
        super(self.__class__, self).__init__()
        self.cache_name = cache_name
        self.sample_file_name = sample_file_name

    def raw_train_samples_gen(self, generator):
        def new_generator():
            for sample in generator:
                yield sample
        return new_generator

    def execute(self, environment={}):
        all_samples = []
        if self.cache_name is None or not os.path.isfile(os.path.join('cache', self.cache_name, self.sample_file_name)):
            for s in environment['raw_train_samples_gen']():
                all_samples.append(s)
                if len(all_samples) % 10000 == 0:
                    print('Cached %i samples' % len(all_samples))
        if self.cache_name is not None:
            file_name = os.path.join('cache', self.cache_name, self.sample_file_name)
            if not os.path.isfile(file_name):
                pickle.dump(all_samples, open(file_name, 'wb'))
            else:
                all_samples = pickle.load(open(file_name, 'rb'))
        environment['raw_train_samples_gen'] = self.raw_train_samples_gen(all_samples)
        return environment

"""
A stemming component expects a text to be a list of tokens. Output are the stemmed words.
"""
class StemmmingFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.fields = fields
        assert isinstance(self.fields,list)

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator


    def transform(self, text, env=None):
        if not isinstance(text, list):
            print ("WARNING!!! THE INPUT TO THE STEMMING COMPONENT IS ASSUMED TO BE A LIST OF WORDS. "
                   "Input seems to be a list of chars")
        return [stem(w) for w in text]


    def execute(self, environment={}):
        np.random.seed(2)
        token_dict = environment['token2id']
        print ('Using stemming!')
        counter = 0

        word_dict_new = {}
        stem2word = {}
        for w in token_dict:
            s_w = stem(w)
            if s_w not in stem2word:
                stem2word[s_w] = counter
                word_dict_new[w] = counter
                counter += 1
            else:
                word_dict_new[w] = stem2word[s_w]
        print ('Stemming reduced token count from %i to %i' %
               (len(token_dict), len(list(set(word_dict_new.values())))))
        environment['token2id'] = word_dict_new
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment


class TrainModelFlowComponent(AbstractFlowComponent):
    def __init__(self,  class_weight=None):
        super(self.__class__, self).__init__()
        self.class_weight = class_weight

    def execute(self, environment={}):
        valid_batch_generator = environment['encoded_valid_samples_gen']
        train_batch_generator = environment['encoded_train_samples_gen']

        # Start the training loop
        print('Starting training...')
        num_valid_samples =int(environment['num_validation_samples'])
        num_batches_pr_epoch = int(environment['config']['sampling']['num_batches_pr_epoch'])
        if num_valid_samples>0:
            environment['model'].fit_generator(generator=train_batch_generator(),
                samples_per_epoch=num_batches_pr_epoch * int(environment['config']['sampling']['batch_size']),
                nb_epoch=int(environment['config']['sampling']['max_epochs']),
                validation_data=valid_batch_generator(),
                nb_val_samples=int(environment['num_validation_samples']),
                # validation_data=environment['valid_samples'],
                nb_worker=1,
                class_weight=self.class_weight,
                # pickle_safe=True,
                callbacks=environment['callbacks'])
        else:
            environment['model'].fit_generator(generator=train_batch_generator(),
               samples_per_epoch=num_batches_pr_epoch * int(environment['config']['sampling']['batch_size']),
               nb_epoch=int(environment['config']['sampling']['max_epochs']),
               callbacks=environment['callbacks'])
        environment['model'].fit = True
        return environment

class SetConfigValFlowComponent(AbstractFlowComponent):
    def __init__(self, key, value):
        super(self.__class__, self).__init__()
        self.key = key
        self.value = value

    def execute(self, environment={}):
        c = environment['config']
        for key in self.key[0:-1]:
            c = c[key]
        c[self.key[-1]] = self.value
        return environment

class SetEnvironmentValueFlowComponent(AbstractFlowComponent):
    def __init__(self, key, value):
        super(self.__class__, self).__init__()
        self.key = key
        self.value = value

    def execute(self, environment={}):
        environment[self.key] = self.value
        return environment

def f(environment, flow_components):
    for component in flow_components:
        print(component.__class__.__name__)
        environment = component.execute(environment)
        assert environment is not None


class FlowController(AbstractFlowComponent):
    def __init__(self, flow_components=[]):
        super(self.__class__, self).__init__()
        self.flow_components = flow_components

    def execute(self, environment={}):
        for component in self.flow_components:
            print (component.__class__.__name__)
            environment = component.execute(environment)
            assert environment is not None
        return environment

    def execute_in_other_process(self, environment={}):
        p = Process(target=f, args=(environment,self.flow_components))
        p.start()
        return environment



class LowercaseFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.fields = fields
        assert isinstance(self.fields,list)

    def transform(self, text, env=None):
        if not isinstance(text, str):
            print ("WARNING!!! THE INPUT TO THE LowercaseFlowComponent IS ASSUMED TO BE A STRING. "
                   "Input seems to be a list of words")
        return text.lower()

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment


class RemovePunktFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, replacement_chars, fields=['text']):
        super(self.__class__, self).__init__()
        self.replacement_chars = replacement_chars
        self.fields = fields
        assert isinstance(self.fields,list)

    def transform(self, text, env=None):
        if not isinstance(text, str):
            print ("WARNING!!! THE INPUT TO THE RemovePunktFlowComponent COMPONENT IS ASSUMED TO BE A STRING.")
            print (text)
            if isinstance(text, list):
                print ("Input seems to be a list of words")
        for c in self.replacement_chars:
            # print (text)
            text = text.replace(c, ' ')
        return text

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment

class DanishNormalizeFlowComponent(AbstractTransformerFlowComponent):
    def generator_generator(self, it):
        def new_generator():
            iterator = it()
            for sample in iterator:
                text = sample['text']
                if len(text) > 0:
                    text = self.transform(text)
                    sample['text'] = text
                yield sample
        return new_generator


    def transform(self, text, env=None):
        text = text.replace('ae', 'æ')
        text = text.replace('oe', 'ø')
        text = text.replace('aa', 'å')
        text = text.replace('Ae', 'Æ')
        text = text.replace('Oe', 'Ø')
        text = text.replace('Aa', 'Å')
        return text



    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment

class PruneInfrequentWords(AbstractFlowComponent):
    def __init__(self, min_token_count):
        super(self.__class__, self).__init__()
        self.min_token_count = min_token_count

    def execute(self, environment={}):
        token2count = environment['token2count']
        token2id_old = environment['token2id']
        token_order = sorted(token2id_old.items(), key=lambda x: x[1])
        token2id = {}
        id_counter = 0
        pruning_count = 0
        for token,_ in token_order:
            count = token2count[token] if token in token2count else 0
            if count >= self.min_token_count:
                token2id[token] = id_counter
                id_counter += 1
            else:
                pruning_count += 1
                # print ('Pruning %s' % token)
        print('Pruned %i tokens' % pruning_count)
        environment['token2id'] = token2id
        return environment


class Token2Id():
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def __len__(self):
        return self.max_tokens

    def values(self):
        return list(range(0, self.max_tokens))

    def __getitem__(self, item):
        v = int(hashlib.sha1(item.encode('utf-8')).hexdigest(), 16)
        return (v % self.max_tokens)

    def __contains__(self, item):
        return isinstance(item, int) and 0 <= item and item < self.max_tokens


class LoadToken2IdHashFlowComponent(AbstractFlowComponent):
    def __init__(self, max_tokens=10**8):
        super(self.__class__, self).__init__()
        self.max_tokens = max_tokens

    def execute(self, environment={}):
        max_tokens = self.max_tokens
        environment['token2id'] = Token2Id(max_tokens)
        return environment




class LoadToken2IdFlowComponent(AbstractFlowComponent):
    def __init__(self, cache_name, fields=['text']):
        super(self.__class__, self).__init__()
        self.cache_name = cache_name
        self.fields = fields
        assert isinstance(self.fields,list)

    def execute(self, environment={}):
        cache_dir = os.path.join(environment['config']['cache']['cache_dir'], self.cache_name)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        filename = os.path.join(cache_dir, 'token2id.pkl')
        if (not os.path.isfile(filename)):
            token2id = {}
            id_counter = 1
            for sample_num, row in enumerate(environment['raw_train_samples_gen']()):
                if sample_num > 0 and sample_num % 10000 == 0:
                    print (sample_num)
                for field in self.fields:
                    text = row[field]
                    for word in text:
                        if word not in token2id:
                            token2id[word] = id_counter
                            id_counter += 1
            pickle.dump(token2id, open(filename, 'wb'))
        else:
            token2id = pickle.load(open(filename, 'rb'))
        environment['token2id'] = token2id
        return environment


class PackageModelFiles(AbstractFlowComponent):
    def __init__(self, package_dir):
        super(self.__class__, self).__init__()
        self.package_dir = package_dir

    def execute(self, environment={}):
        if not os.path.isdir(self.package_dir):
            print ('Creating directory: ' + self.package_dir)
            os.makedirs(self.package_dir)
        dependencies = sum([t[1] for t in environment['transformations']], []) + ['config']
        if 'dependencies' in environment:
            print ('Adding dependencies to environment %s' % str(environment['dependencies']))
            dependencies += environment['dependencies']
        dependencies = list(set(dependencies))
        env = dict([(key, environment[key]) for key in dependencies])
        pickle.dump((env,
                     environment['transformations'],
                     environment['model_names']),
                    open(os.path.join(self.package_dir, 'environment.pkl'), 'wb'),protocol=4)
        for model_name in environment['model_names']:
            parts = model_name.split('.')
            shutil.copy(os.path.join(model_name.replace('.', '/')+'.py'),
                    os.path.join(self.package_dir, parts[-1] + '.py'))
        open(os.path.join(self.package_dir,'__init__.py'), 'a').close()
        return environment

class PrintNumTokensFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()

    def execute(self, environment={}):
        print ('Number of words in model: %i ' % len(environment['token2id']))
        return environment

class PrintNumClassesFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()

    def execute(self, environment={}):
        print ('Number of classes in model: %i ' % len(environment['class2id']))
        return environment

class LoadClass2IdFlowComponent(AbstractFlowComponent):
    def __init__(self, cache_name):
        super(self.__class__, self).__init__()
        self.cache_name=cache_name

    def execute(self, environment={}):
        cache_dir = os.path.join(environment['config']['cache']['cache_dir'], self.cache_name)
        class2id_file = os.path.join(cache_dir, 'class2id.pkl')
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        if os.path.isfile(class2id_file):
            class2id = pickle.load(open(class2id_file, 'rb'))
        else:
            if 'class2samples' in environment:
                print('Using class2smaples for class2id')
                classes = list(environment['class2samples'].keys())
                class2id = dict((c, i) for i, c in enumerate(classes))
            else:
                class2id = {}
                counter = 0
                for i, sample in enumerate(environment['raw_train_samples_gen']()):
                    class_name = sample['class_']
                    if class_name not in class2id:
                        class2id[class_name] = counter
                        counter+=1
            pickle.dump(class2id, open(class2id_file, 'wb'))
        environment['class2id'] = class2id
        environment.setdefault('dependencies', [])
        environment['dependencies'].append('class2id')
        return environment

class PrintNumDocsFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()

    def execute(self, environment={}):
        document_counter = 0
        for _ in (environment['raw_train_samples_gen']()):
            document_counter += 1
        print ('Number of documents in corpus: %i ' % document_counter)
        return environment


class SnippetTrainGenerator(AbstractTransformerFlowComponent):
    @abc.abstractmethod
    def create_snippet(self, snippet_size):
        pass

class SynonymInjectionSnippetGeneratorFlowComponent(SnippetTrainGenerator):
    def __init__(self, synonym_file, injection_probability=0.5, verbose=False):
        super(self.__class__, self).__init__()
        self.synonym_file = synonym_file
        self.lines = open(self.synonym_file, 'r').read().split('\n')
        # self.lines = self.lines[0:20]
        self.injection_probability = injection_probability
        self.verbose = verbose


    def create_snippet(self, snippet_size):
        snippet, c = self.prev_snippet_gen.create_snippet(snippet_size)
        new_snippet = []
        substitution = False
        for w in snippet:
            if w not in self.word2synonym_idx \
                    or np.random.random() > self.injection_probability \
                    or len(self.environment['id2tokens'][w])>1:
                new_snippet.append(w)
                continue
            substitution = True
            substitutions = self.lines[self.word2synonym_idx[w]]
            sub = np.random.choice(substitutions)
            new_snippet.append(sub)
        # print(self.id2tokens)
        if substitution and self.verbose:
            id2tokens = self.environment['id2tokens']
            a = ' '.join(['|'.join(id2tokens[l]) if l in id2tokens else 'UNKNOWN' for l in snippet ])
            b = ' '.join(['|'.join(id2tokens[l]) if l in id2tokens else 'UNKNOWN' for l in new_snippet ])
            if 'UNKNOWN' not in b and not a == b:
                print(a)
                print(b)
                print('')
                # print('%s --> %s' % (a,b))
        return new_snippet,c

    def execute(self, environment={}):
        from flow_framework.environment_wrapper import EnvironmentWrapper
        encoder = EnvironmentWrapper(environment)
        new_lines = []
        self.word2synonym_idx = {}
        self.environment = environment
        for idx, line in enumerate(self.lines):
            words = line.split(',')
            if len(line) == 0:
                continue
            words = [encoder.encode(w.strip())[0] for w in words]
            new_lines.append(words)
            for w in words:
                self.word2synonym_idx[w] = idx
        self.lines = new_lines
        self.prev_snippet_gen = environment['raw_snippet_gen']
        environment['raw_snippet_gen'] = self
        return environment


class ToSetSnippetGeneratorFlowComponent(SnippetTrainGenerator):
    def __init__(self):
        super(self.__class__, self).__init__()


    def create_snippet(self, snippet_size):
        snippet, c = self.prev_snippet_gen.create_snippet(snippet_size)
        new_snippet = list(set(snippet))
        return new_snippet,c

    def execute(self, environment={}):
        self.prev_snippet_gen = environment['raw_snippet_gen']
        environment['raw_snippet_gen'] = self
        self.environment = environment
        return environment



class HashingSnippetGeneratorsFlowComponent(SnippetTrainGenerator):
    def __init__(self):
        super(self.__class__, self).__init__()

    def create_snippet(self, snippet_size):
        snippet, c = self.prev_snippet_gen.create_snippet(snippet_size)
        snippet = [self.token2id[n] if n in self.token2id else 0 for n in snippet]
        return snippet,c

    def execute(self, environment={}):
        self.token2id = environment['token2id']
        self.prev_snippet_gen = environment['raw_snippet_gen']
        environment['raw_snippet_gen'] = self
        return environment


class OnlineTrainSamplesSnippetGeneratorsFlowComponent(SnippetTrainGenerator):
    def __init__(self, randomize=False):
        super(self.__class__, self).__init__()
        self.randomize=randomize

    def create_snippet(self, snippet_size):
        sample = next(self.sample_gen)
        snippet = self.random_substring(sample['text'], snippet_size), sample['class_']
        return snippet

    def execute(self, environment={}):
        def gen():
            while True:
                for sample in environment['raw_train_samples_gen']():
                    yield sample
        self.sample_gen = gen()
        environment['raw_snippet_gen'] = self
        return environment


    def random_substring(self, text, length=100):
        """Pick out a random substring of a certain length.

        If the original string is shorter than the desired length of the
        substring, the whole string is returned.
        """
        if self.randomize:
            np.random.shuffle(text)
        string_length = len(text)
        if string_length <= length:
            ret_string = text
        else:
            max_start = string_length - length
            start = random.randint(0, max_start)
            ret_string = text[start:start + length]
        return ret_string

class TrainSamplesSnippetGeneratorsFlowComponent(SnippetTrainGenerator):
    def __init__(self, randomize=False):
        super(self.__class__, self).__init__()
        self.randomize=randomize

    def create_snippet(self, snippet_size):
        key = np.random.choice(self.keys, 1,p=self.weights)[0]
        texts = self.class2samples[key]
        num_texts = len(texts)
        text_num = np.random.randint(0,num_texts,1)[0]
        text = texts[text_num]
        snippet = self.random_substring(text, snippet_size), key
        return snippet

    def execute(self, environment={}):
        self.class2samples = environment['class2samples']
        self.keys = list(self.class2samples.keys())
        # self.weights = np.asarray([len(self.class2samples[key]) for key in self.class2samples],dtype='float32')
        self.weights = np.asarray([1]*len(self.class2samples),dtype='float32')
        self.weights/= np.sum(self.weights)
        environment['raw_snippet_gen'] = self
        return environment

    def random_substring(self, text, length=100):
        """Pick out a random substring of a certain length.

        If the original string is shorter than the desired length of the
        substring, the whole string is returned.
        """
        if self.randomize:
            np.random.shuffle(text)
        string_length = len(text)
        if string_length <= length:
            ret_string = text
        else:
            max_start = string_length - length
            start = random.randint(0, max_start)
            ret_string = text[start:start + length]
        return ret_string


class UniformMemoryEfficientSnippetGeneratorsFlowComponent(SnippetTrainGenerator):
    def __init__(self, randomize=False, dict_size=200000, num_passes=5):
        super(self.__class__, self).__init__()
        self.randomize=randomize
        self.dict_size = dict_size
        self.num_passes = num_passes

    def text_generator(self, train_samples_gen):
        class2samples = {}
        sample_count = 0
        while True:
            for i, sample in enumerate(train_samples_gen()):
                class_name = sample['class_']
                class2samples.setdefault(class_name, [])
                class2samples[class_name].append(sample)
                sample_count += 1
                if sample_count % (self.dict_size) == 0:
                    keys = list(class2samples.keys())
                    for _ in range(0, sample_count*self.num_passes):
                        key = np.random.choice(keys)
                        samples = class2samples[key]
                        sample = samples[np.random.randint(0, len(samples))]
                        yield sample
                    class2samples = {}
                    sample_count = 0


    def create_snippet(self, snippet_size):
        sample = next(self.train_samples_gen)
        key = sample['class_']
        text = sample['text']
        if self.randomize:
            np.random.shuffle(text)
        snippet = self.random_substring(text, snippet_size), key
        return snippet

    def execute(self, environment={}):
        total_count = np.sum(list(environment['class2count'].values()))
        self.dict_size = min(self.dict_size, total_count)

        # self.class2probs = dict([(name, float(total_count)/count) for (name, count) in environment['class2count'].items()])
        # total_count = np.sum(list(self.class2probs.values()))
        # self.class2probs = dict([(name, float(count) / total_count) for (name, count) in self.class2probs.items()])

        self.train_samples_gen = self.text_generator(environment['raw_train_samples_gen'])
        environment['raw_snippet_gen'] = self
        return environment

    def random_substring(self, text, length=100):
        """Pick out a random substring of a certain length.

        If the original string is shorter than the desired length of the
        substring, the whole string is returned.
        """
        if self.randomize:
            np.random.shuffle(text)
        string_length = len(text)
        if string_length <= length:
            ret_string = text
        else:
            max_start = string_length - length
            start = random.randint(0, max_start)
            ret_string = text[start:start + length]
        return ret_string






class CachedUniformMemoryEfficientSnippetGeneratorsFlowComponent(SnippetTrainGenerator):
    def __init__(self, cache_name, max_samples_in_memory, randomize=False, save_every=200000, num_passes =1, encode=None):
        super(self.__class__, self).__init__()
        self.randomize=randomize
        self.save_every = save_every
        self.num_passes = num_passes
        self.cache_name = cache_name
        self.max_samples_in_memory = max_samples_in_memory
        if encode is None:
            self.encode = lambda x: x
        else:
            self.encode = encode

    def text_generator(self):
        while True:
            sample_count = 0
            class2samples = {}
            files = os.listdir(self.cache_dir)
            samples_pr_class = int(float(self.max_samples_in_memory)/len(files))
            t0 = time.clock()
            for file in files:
                samples = pickle.load(open(os.path.join(self.cache_dir, file), 'rb'))
                np.random.shuffle(samples)
                samples = samples[0:samples_pr_class]
                class2samples[file] = samples
                sample_count += len(samples)
            print ('Loaded classes in %1.2f seconds' % (time.clock()-t0))
            keys = list(class2samples.keys())
            for _ in range(0, sample_count * self.num_passes):
                key = np.random.choice(keys)
                samples = class2samples[key]
                sample = samples[np.random.randint(0, len(samples))]
                yield sample

    def create_snippet(self, snippet_size):
        sample = next(self.train_samples_gen)
        key = sample['class_']
        text = sample['text']
        if self.randomize:
            np.random.shuffle(text)
        snippet = self.random_substring(text, snippet_size), key
        return snippet

    def execute(self, environment={}):
        self.cache_dir = os.path.join(environment['config']['cache']['cache_dir'], self.cache_name, 'classes')
        cached_data_exists = True
        if not os.path.isdir(self.cache_dir) or len(os.listdir(self.cache_dir))==0:
            cached_data_exists = False

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        filename = os.path.join(self.cache_dir, '%s')
        class2samples = {}

        def save():
            for c in class2samples:
                current_file = filename % c
                samples = class2samples[c]
                if os.path.exists(current_file):
                    samples += pickle.load(open(current_file, 'rb'))
                pickle.dump(list(samples), open(current_file, 'wb'))

        if not cached_data_exists:
            print('Creating cached class2samples...')
            sample_count = 0
            for i, sample in enumerate(environment['raw_train_samples_gen']()):
                class_name = sample['class_']
                class2samples.setdefault(class_name, [])
                sample = self.encode(sample, environment)
                class2samples[class_name].append(sample)
                sample_count += 1
                if sample_count % 10000 == 0:
                    print ('%i\tCachedUniformMemoryEfficientSnippetGeneratorsFlowComponent'%(sample_count))
                if sample_count % self.save_every == 0:
                    save()
                    class2samples = {}
            save()
            class2samples = {}

            print('Done creating cached class2samples')

        environment['raw_snippet_gen'] = self
        self.train_samples_gen = self.text_generator()

        return environment

    def random_substring(self, text, length=100):
        """Pick out a random substring of a certain length.

        If the original string is shorter than the desired length of the
        substring, the whole string is returned.
        """
        if self.randomize:
            np.random.shuffle(text)
        string_length = len(text)
        if string_length <= length:
            ret_string = text
        else:
            max_start = string_length - length
            start = random.randint(0, max_start)
            ret_string = text[start:start + length]
        return ret_string



class CachedTrainGeneratorsFlowComponent(AbstractFlowComponent):
    def __init__(self, verbose=False, max_size=np.inf, refresh_cache_every=5):
        super(self.__class__, self).__init__()
        self.verbose = verbose
        self.max_size = max_size
        self.refresh_cache_every = refresh_cache_every
        self.iteration_num = 0

    def read_train_samples(self):
        self.cache = []
        print('Reading samples into cache...')
        for sample in self.train_gen():
            self.cache.append(sample)
            if self.verbose:
                if len(self.cache)%10000 == 0:
                    print (len(self.cache))
            if self.max_size <= len(self.cache):
                break

    def generator_generator(self):
        def new_generator():
            if self.iteration_num % self.refresh_cache_every == 0:
                self.read_train_samples()
            np.random.shuffle(self.cache)
            for sample in self.cache:
                yield sample
            self.iteration_num += 1
        return new_generator

    def execute(self, environment={}):
        self.train_gen =  environment['raw_train_samples_gen']
        environment['raw_train_samples_gen'] = self.generator_generator()
        return environment



def get_n_grams(words, n):
    n_grams = []
    for i in range(0, len(words) - n + 1):
        n_grams.append(' '.join(words[i:(i + n)]))
    return n_grams

def get_all_n_grams(words, max_n):
    all_n_grams = sum([get_n_grams(words, n) for n in range(1, max_n)], [])
    return all_n_grams

class LoadUmlsDatabaseFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()

    def execute(self, environment={}):
        filename = 'cui2info.pkl'
        database_dir = environment['config']['umls']['database_dir']
        full_path = os.path.join(database_dir, filename)
        if not os.path.isdir(database_dir):
            os.makedirs(database_dir)
        if not os.path.isfile(full_path):
            print('downloading cui2info mapping') # SEE DATA_SCRIPTS folder for how to create cui2info!
            urllib.request.urlretrieve('http://www.intellifind.dk/umls/cui2info.pkl', full_path)
        sem_typ2sem_group, cui2info = pickle.load(open(full_path, 'rb'))
        environment['cui2info'] = cui2info
        environment['sem_typ2sem_group'] = sem_typ2sem_group
        return environment


class FreezeModelFlowComponent(AbstractFlowComponent):
    def __init__(self, layer_names=None):
        super(self.__class__, self).__init__()
        self.layer_names = layer_names

    def execute(self, environment={}):
        layers = environment['model'].layers
        for layer in layers:
            if self.layer_names is not None:
                if layer.name in self.layer_names:
                    layer.trainable_weights = []
            else:
                layer.trainable_weights = []
        return environment




class UMLSTransformFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.stopwords = ['is', 'was', 'pap', 'go', 'a', 'at', 'an', 'of', 'in', 'as', 'all', 'on', 'unknown', 'other',
                          'can']
        self.fields = fields
        assert isinstance(self.fields,list)

    def transform(self, text, env=None):
        if not isinstance(text, list):
            print(
                "WARNING!!! THE INPUT TO THE UMLSSnippetGeneratorFlowComponent COMPONENT IS ASSUMED TO BE A list of tokens.")
        ngrams = sum([get_n_grams(text, n) for n in range(1, 5)], [])
        return sum([self.all_names[n] for n in ngrams
                    if n in self.all_names and n not in self.stopwords and not n.isnumeric() and len(n) > 1], [])

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample

        return new_generator

    def execute(self, environment={}):
        cui2info = environment['cui2info']
        sem_typ2sem_group = environment['sem_typ2sem_group']
        # all_names = []
        all_names_dict = {}
        for cui, info in cui2info.items():
            sem_types = info['sem_typ']
            if len([s for s in sem_types if
                    sem_typ2sem_group[s.lower()] in ['disorders', 'genes & molecular sequences']]) == 0:
                continue
            for name in info['aliases']:
                name = name.lower()
                all_names_dict.setdefault(name, [])
                all_names_dict[name].append(cui.lower())
                # all_names += [(name.lower(), cui.lower()) for name in info['aliases']]
        # self.all_names = dict(all_names)


        self.all_names = dict((n, list(set(all_names_dict[n]))) for n in all_names_dict)
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment


class UpperLowerGeneratorFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.fields = fields
        assert isinstance(self.fields,list)


    def transform(self, text, env=None):
        if not isinstance(text, list):
            print ("WARNING!!! THE INPUT TO THE UpperLowerGeneratorFlowComponent COMPONENT IS ASSUMED TO BE A list of tokens.")
        new_text = []
        for s in text:
            if s.lower() != s:
                new_text += [s.lower()]
            new_text.append(s)
        return new_text

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment


class SpellingErrorFlowComponent(AbstractFlowComponent):
    def __init__(self, addition_percentage=0.05, removal_percentage=0.05, fields=['text']):
        super(self.__class__, self).__init__()
        self.addition_percentage = addition_percentage
        self.removal_percentage = removal_percentage
        self.fields = fields
        assert isinstance(self.fields,list)

    def transform(self, text, env=None):
        if not isinstance(text, list):
            print ("WARNING!!! THE INPUT TO THE SpellingErrorFlowComponent COMPONENT IS ASSUMED TO BE A list of tokens.")
        new_text = []
        for t in text:
            p = np.random.random(size=2)
            if p[0] < self.addition_percentage:
                new_text.append(np.random.choice(np.asarray(list(string.printable))))
            if p[1] > self.removal_percentage: # do not remove token
                new_text.append(t)
        return new_text

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        return environment



class UniformSamplingFlowComponent(AbstractFlowComponent):
    def __init__(self, addition_percentage=0.05, removal_percentage=0.05, fields=['text']):
        super(self.__class__, self).__init__()
        self.addition_percentage = addition_percentage
        self.removal_percentage = removal_percentage
        self.fields = fields
        assert isinstance(self.fields,list)

    def transform(self, text, env=None):
        if not isinstance(text, list):
            print ("WARNING!!! THE INPUT TO THE SpellingErrorFlowComponent COMPONENT IS ASSUMED TO BE A list of tokens.")
        new_text = []
        for t in text:
            p = np.random.random(size=2)
            if p[0] < self.addition_percentage:
                new_text.append(np.random.choice(np.asarray(list(string.printable))))
            if p[1] > self.removal_percentage: # do not remove token
                new_text.append(t)
        return new_text

    def generator_generator(self, it):

        def new_generator():

            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                yield sample
        return new_generator

    def execute(self, environment={}):
        self.class2samples = {}
        for i, sample in enumerate(environment['raw_train_samples_gen']()):
            class_name = sample['class_']
            self.class2samples.setdefault(class_name, [])
            self.class2samples[class_name].append(sample)
            if i % 10000 == 0:
                print('%i\tUniformSamplingFlowComponent' %i)
        self.keys = list(self.class2samples.keys())
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        return environment




class BreakNgramSnippetGeneratorFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text']):
        super(self.__class__, self).__init__()
        self.fields = fields
        assert isinstance(self.fields,list)


    def transform(self, text, env=None):
        if not isinstance(text, list):
            print ("WARNING!!! THE INPUT TO THE BreakNgramSnippetGeneratorFlowComponent COMPONENT IS ASSUMED TO BE A list of tokens.")
        new_text = []
        for s in text:
            if '_' in s:
                new_text += s.split('_')
            new_text.append(s)
        return new_text

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field])
                # print (sample['text'][0:100])
                yield sample
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment



class JoinFieldsFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, fields=['text', 'title'], target_field='text'):
        super(self.__class__, self).__init__()
        self.fields = fields
        self.target_field = target_field
        assert isinstance(self.fields,list)


    def transform(self, sample, env=None):
        sample[self.target_field] = ' '.join([sample[field] for field in self.fields if field in sample])
        return sample

    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                sample = self.transform(sample)
                yield sample
        return new_generator

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'])
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'])
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'])
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, [], self.__class__.__name__))
        return environment



class PrintNumSamplesFlowComponent(AbstractFlowComponent):
    def __init__(self, targets=['train', 'test', 'valid']):
        super(self.__class__, self).__init__()
        self.targets = targets



    def generator_generator(self, it):
        def new_generator():
            for sample in it():
                sample = self.transform(sample)
                yield sample
        return new_generator

    def execute(self, environment={}):
        print('Counting samples...')
        for target in self.targets:
            count = 0
            for s in environment['raw_%s_samples_gen'%target]():
                count += 1
            print ('%s: %i' % (target, count))
        return environment



class ExpandNgramsIdFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, cache_name, min_ngram_count=5, max_n=10, max_num_ngrams=2*(10**6), pruning_limit=(2*(10 ** 7)), randomize=True, fields=['text']):
        super(self.__class__, self).__init__()
        self.cache_name = cache_name
        self.min_ngram_count = min_ngram_count
        self.max_n = max_n
        self.max_num_ngrams = max_num_ngrams
        self.randomize = randomize
        self.fields = fields
        self.pruning_limit = pruning_limit
        assert isinstance(self.fields,list)

    def execute(self, environment={}):
        cache_dir = os.path.join(environment['config']['cache']['cache_dir'], self.cache_name)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        filename = os.path.join(cache_dir, 'all_ngrams.pkl')
        if (not os.path.isfile(filename)):
            print('Creating all_ngrams.pkl...')
            ngrams = {}
            for count, s in enumerate(environment['raw_train_samples_gen']()):
                if count % 10000 == 0:
                    print (count)
                for field in self.fields:
                    text = s[field]
                    for n in range(1, self.max_n+1):
                        for i in range(0, len(text) - 1):
                            ngram = '_'.join(text[i:(i + n)])
                            ngrams.setdefault(ngram, 0)
                            ngrams[ngram] += 1
                    if len(ngrams) > self.pruning_limit:
                        ngrams_sort = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
                        ngrams = dict([(g, c) for (g, c) in ngrams_sort if c > 1][0:(self.max_num_ngrams*2)])
                        print('ExpandNgramsIdFlowComponent: num_docs=%i, num_ngrams=%i'%(count, len(ngrams)))
            ngrams_sort = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
            ngrams = dict([(g, c) for (g, c) in ngrams_sort if c >= self.min_ngram_count][0:self.max_num_ngrams])
            pickle.dump(ngrams, open(filename, 'wb'))
            print('Saving all_ngrams.pkl...')
        else:
            ngrams = pickle.load(open(filename, 'rb'))
        print('ExpandNgramsIdFlowComponent: num_ngrams=%i' % (len(ngrams)))
        environment['all_ngrams'] = ngrams

        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'], environment)
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'], environment)
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'], environment)
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, self.get_dependencies(), self.__class__.__name__))
        return environment


    def transform(self, text, env=None):
        if not isinstance(text, list):
            print(
                "WARNING!!! THE INPUT TO THE ExpandNgramsIdFlowComponent COMPONENT IS ASSUMED TO BE A list of tokens.")
        all_ngrams = env['all_ngrams']
        new_text = []
        for n in range(1, self.max_n+1):
            for i in range(0, len(text)):
                ngram = '_'.join(text[i:(i + n)])
                if ngram in all_ngrams:
                    new_text.append(ngram)
        # new_text = list(set(new_text))
        if self.randomize:
            np.random.shuffle(new_text)
        return new_text


    def generator_generator(self, it, env):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field], env)
                yield sample
        return new_generator


    def get_dependencies(self):
        return ['all_ngrams']


class ExpandAllNgramsFlowComponent(AbstractTransformerFlowComponent):
    def __init__(self, max_n=10, randomize=True, to_set=True, fields=['text']):
        super(self.__class__, self).__init__()
        self.max_n = max_n
        self.randomize = randomize
        self.fields = fields
        self.to_set = to_set
        assert isinstance(self.fields,list)

    def execute(self, environment={}):
        environment['raw_train_samples_gen'] = self.generator_generator(environment['raw_train_samples_gen'], environment)
        environment['raw_valid_samples_gen'] = self.generator_generator(environment['raw_valid_samples_gen'], environment)
        environment['raw_test_samples_gen'] = self.generator_generator(environment['raw_test_samples_gen'], environment)
        if 'transformations' not in environment:
            environment['transformations'] = []
        environment['transformations'].append((self.transform, self.get_dependencies(), self.__class__.__name__))
        return environment


    def transform(self, text, env=None):
        if not isinstance(text, list):
            print(
                "WARNING!!! THE INPUT TO THE ExpandNgramsIdFlowComponent COMPONENT IS ASSUMED TO BE A list of tokens.")
        new_text = []
        for i in range(0, len(text)):
            for n in range(1, self.max_n+1):
                ngram = '_'.join(text[i:(i + n)])
                new_text.append(ngram)
        if self.to_set:
            new_text = list(set(new_text))
        if self.randomize:
            np.random.shuffle(new_text)
        return new_text


    def generator_generator(self, it, env):
        def new_generator():
            for sample in it():
                for field in self.fields:
                    sample[field] = self.transform(sample[field], env)
                yield sample
        return new_generator


    def get_dependencies(self):
        # return ['all_ngrams']
        return []