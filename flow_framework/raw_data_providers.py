import nltk
import pandas as pd
import os
import pickle
from flow_framework.base_provider import BaseProvider
from flow_framework import get_n_grams
from zipfile import ZipFile
import csv
import lxml.html as lh
from lxml.html.clean import clean_html
import re
import urllib.request
import random
import numpy as np
import tarfile

csv.field_size_limit(10**9)

class BordingProvider(BaseProvider):
    def raw_train_samples_gen(self):
        for row_num, sample in self.corpus.iterrows():
            sample['class_'] = sample['cui'].lower()
            yield sample

    def raw_valid_samples_gen(self):
        for cui, text in self.valid_samples:
            sample = {'class_': cui.lower(), 'text': text}
            yield sample

    def raw_test_samples_gen(self):
        for cui, text in self.test_samples:
            sample = {'class_': cui.lower(), 'text': text}
            yield sample

    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        corpus_file = os.path.join(data_dir, 'corpus.csv')
        cui2idx_file = os.path.join(data_dir, 'cui2idx.csv')
        test_queries = os.path.join(data_dir, 'test_queries.csv')
        validation_queries = os.path.join(data_dir, 'valid_queries.csv')
        if not os.path.exists(data_dir):
            print('Creating data directory')
            os.makedirs('data/bording')
        if not os.path.isfile(corpus_file):
            print('downloading training data (~140 MB)')
            urllib.request.urlretrieve('https://www.dropbox.com/s/ty8bwft5ttpuyzx/corpus.csv?dl=1', corpus_file)
        if not os.path.isfile(cui2idx_file):
            print('downloading other data')
            urllib.request.urlretrieve('https://www.dropbox.com/s/03lzqdhkn6abfw1/cui2idx.csv?dl=1', cui2idx_file)
        if not os.path.isfile(validation_queries):
            print('downloading validation data')
            urllib.request.urlretrieve('http://www.intellifind.dk/doctors_dilemma/validation_queries.csv', validation_queries)
        if not os.path.isfile(test_queries):
            print('downloading test queries')
            urllib.request.urlretrieve('http://www.intellifind.dk/doctors_dilemma/test_queries.csv', test_queries)
            print ('Data download complete')

            print('Done')

    def load_data(self):
        self.environment['class2id'] = pd.read_csv(os.path.join(self.environment['config']['data']['data_dir'], 'cui2idx.csv'), index_col=0,encoding='utf-8').to_dict()['0']
        # self.val_set = pd.read_csv(os.path.join(self.environment['config']['data']['data_dir'], 'validation_queries.csv'),encoding='utf-8')
        self.corpus = pd.read_csv(os.path.join(self.environment['config']['data']['data_dir'], 'corpus.csv'),encoding='utf-8')

        with open(os.path.join(self.environment['config']['data']['data_dir'], 'valid_queries.csv'), 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.valid_samples = [(row[2].lower(), row[1]) for row in reader][1::]
            self.environment['num_validation_samples'] = len(self.valid_samples )

        with open(os.path.join(self.environment['config']['data']['data_dir'], 'test_queries.csv'), 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.test_samples = [(row[2].lower(), row[1]) for row in reader][1::]
            self.environment['num_test_samples'] = len(self.test_samples)






import tarfile
from random import shuffle
from math import floor as fl
from copy import deepcopy
"""
Raw data iterator for the dbpedia dataset
"""
class DBPediaProvider(BaseProvider):
    def __init__(self, max_test_samples=None):
        self.max_test_samples = max_test_samples

    def raw_train_samples_gen(self):
        # for sample in self.train_samples[0:100]:
        for sample in self.train_samples:
            yield deepcopy(sample)

    def raw_valid_samples_gen(self):
        # for sample in self.valid_samples[0:100]:
        for sample in self.valid_samples:
            yield deepcopy(sample)

    def raw_test_samples_gen(self):
        for sample in self.test_samples:
            yield deepcopy(sample)

    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        corpus_file = os.path.join(data_dir, 'dbpedia_csv.tar.gz')
        if not os.path.exists(data_dir):
            print('creating data directory')
            os.makedirs(data_dir)

            print('downloading data (~67 MB)')
            urllib.request.urlretrieve('http://www.intellifind.dk/dbpedia_csv.tar.gz',
                                       corpus_file)
        if not os.path.exists(os.path.join(data_dir, 'dbpedia_csv')):
            tar = tarfile.open(corpus_file, 'r')
            tar.extractall(data_dir)
            tar.close()

    def load_data(self):
        valid_split = float(self.environment['config']['data']['valid_split'])
        data_dir = os.path.join(self.environment['config']['data']['data_dir'], 'dbpedia_csv')
        train_file = os.path.join(data_dir, 'train.csv')
        test_file = os.path.join(data_dir, 'test.csv')
        classes_file = os.path.join(data_dir, 'classes.txt')
        classes = open(classes_file).read().split('\n')[0:14]
        self.environment['class2id'] = dict([(c,i+1) for i, c in enumerate(classes)])
        self.environment['id2class'] = dict([(i+1, c) for i, c in enumerate(classes)])
        train_samples_reader = csv.DictReader(open(train_file, encoding='utf-8'),
                                              fieldnames=['class', 'title', 'text'],
                                              dialect='unix')
        test_samples_reader = csv.DictReader(open(test_file, encoding='utf-8'),
                                             fieldnames=['class', 'title', 'text'],
                                             dialect='unix')
        train_samples_all = []
        self.test_samples = []
        for row in train_samples_reader:
            row['class_'] = self.environment['id2class'][int(row['class'])]
            train_samples_all.append(row)

        for row in test_samples_reader:
            row['class_'] = self.environment['id2class'][int(row['class'])]
            self.test_samples.append(row)
        random.seed(1)
        random.shuffle(train_samples_all)
        random.shuffle(self.test_samples)
        if self.max_test_samples is not None:
            self.test_samples = self.test_samples[0:self.max_test_samples]
        n = len(train_samples_all)
        split_idx = fl(n*valid_split)
        self.valid_samples = train_samples_all[0:split_idx]
        self.train_samples = train_samples_all[split_idx::]
        self.environment['num_validation_samples'] = len(self.valid_samples)


"""
Raw data iterator for the opensubtitles dataset
"""
class OpenSubtitlesProvider(BaseProvider):
    def __init__(self, language_pair='en-fr', num_valid_samples=2500, test_fraction=0.05):
        self.zip_file_name = '%s.txt.zip' % language_pair
        self.source_url = 'http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/%s' % self.zip_file_name
        self.test_fraction = test_fraction
        self.num_valid_samples = num_valid_samples
        self.languages = language_pair.split('-')

    def raw_train_samples_gen(self):
        for target, (categories, texts) in self.train_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def raw_valid_samples_gen(self):
        for target, (categories, texts) in self.valid_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def raw_test_samples_gen(self):
        for target, (categories, texts) in self.test_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        corpus_file = os.path.join(data_dir, self.zip_file_name)

        if not os.path.exists(data_dir):
            print('creating data directory: %s' % self.environment['config']['data']['data_dir'])
            os.makedirs(self.environment['config']['data']['data_dir'])
        if not os.path.isfile(corpus_file):
            print('Downloading training data (~5 MB)')
            urllib.request.urlretrieve(self.source_url, corpus_file)
            print ('Data download complete')

    def load_data(self):
        with ZipFile(os.path.join(self.environment['config']['data']['data_dir'], self.zip_file_name), 'r') as ziparchive:
            self.train_data = []
            self.language_order = []
            np.random.seed(1)
            num_lines_in_sample = 2

            self.train_samples = []
            for count, filename in enumerate(ziparchive.namelist()[0:]):
                lang = filename.split('.')[-1]
                self.language_order.append(lang)
                with ziparchive.open(filename, 'r') as f:
                    lines = f.read().decode('utf-8').splitlines()
                    lines = ['. '.join(lines[i:(i+num_lines_in_sample)]) for i in range(0, len(lines)-num_lines_in_sample,num_lines_in_sample)]
                    # lines = [' '.join(lines[i:(i+num_lines_in_sample)]) for i in range(0, len(lines), num_lines_in_sample)]
                    self.train_data.append(zip([lang]*len(lines), lines))
            corpus = zip(*self.train_data)
            for i, c in enumerate(list(corpus)):
                # s = str(i), (['N/A'], list(c))
                s = list(c)[0][1], (['N/A'], list(c))
                self.train_samples.append(s)

            num_samples = 0
            for _, (_, texts) in self.train_samples:
                num_samples += len(texts)
            print('Num samples: %i' % num_samples)

            np.random.seed(1)
            np.random.shuffle(self.train_samples)
            num_samples = len(self.train_samples)
            # num_valid_samples = int(num_samples * self.num_valid_samples)
            num_test_samples = int(num_samples * self.test_fraction)
            self.valid_samples = self.train_samples[0:self.num_valid_samples]
            self.test_samples = self.train_samples[self.num_valid_samples:(self.num_valid_samples + num_test_samples)]
            self.train_samples = self.train_samples[(self.num_valid_samples + num_test_samples)::]

            self.environment['num_validation_samples'] = len(self.valid_samples)
            self.environment['num_test_samples'] = len(self.test_samples)


"""
Raw data iterator for the europarl dataset
"""
class EuroparlProvider(BaseProvider):
    def __init__(self, languages = ['da'], num_valid_samples=2500, test_fraction=0.05, num_lines_in_sample=3):
        # self.pickle_file_name = 'new_dict_en_da_fr.pkl'
        self.languages = languages
        self.num_valid_samples = num_valid_samples
        self.test_fraction = test_fraction
        self.num_lines_in_sample = num_lines_in_sample

    def raw_train_samples_gen(self):
        for target, (categories, texts) in self.train_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def raw_valid_samples_gen(self):
        for target, (categories, texts) in self.valid_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def raw_test_samples_gen(self):
        for target, (categories, texts) in self.test_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        if not os.path.exists(data_dir):
            print('creating data directory: %s' % self.environment['config']['data']['data_dir'])
            os.makedirs(self.environment['config']['data']['data_dir'])


        for lang in self.languages:
            corpus_file1 = os.path.join(data_dir, 'europarl-v7.%s-en.%s' % (lang, lang))
            corpus_file2 = os.path.join(data_dir, 'europarl-v7.%s-en.%s' % (lang, 'en'))

            tar_file = os.path.join(data_dir, '%s.tar' % lang)


            if not os.path.isfile(corpus_file1) or not os.path.isfile(corpus_file2):
                print('Downloading training data (~500 MB)')
                url = 'http://www.statmt.org/europarl/v7/%s-en.tgz' % lang
                urllib.request.urlretrieve(url, tar_file)
                print('Data download complete')
                print('Extracting %s language files...' % lang)
                tar = tarfile.open(tar_file)
                tar.extractall(data_dir)
                tar.close()
                os.remove(tar_file)


    def load_data(self):
        data_dir = self.environment['config']['data']['data_dir']
        corpus = []
        self.train_samples = []
        for lang in self.languages:
            corpus_file1 = os.path.join(data_dir, 'europarl-v7.%s-en.%s' % (lang, lang))
            corpus_file2 = os.path.join(data_dir, 'europarl-v7.%s-en.%s' % (lang, 'en'))
            data1 = open(corpus_file1, encoding='utf-8').readlines()
            data2 = open(corpus_file2, encoding='utf-8').readlines()

            data1 = ['. '.join(data1[i:(i+self.num_lines_in_sample)]) for i in range(0, len(data1)-self.num_lines_in_sample,self.num_lines_in_sample)]
            data2 = ['. '.join(data2[i:(i+self.num_lines_in_sample)]) for i in range(0, len(data2)-self.num_lines_in_sample,self.num_lines_in_sample)]

            data1 = zip([lang]*len(data1), data1)
            data2 = zip(['en']*len(data2), data2)
            corpus += zip(data1, data2)
        for i, c in enumerate(corpus):
            s = list(c)[0][1], (['N/A'], list(c))
            # s = i, (['N/A'], list(c))
            self.train_samples.append(s)

        num_samples = 0
        for _, (_, texts) in self.train_samples:
            num_samples += len(texts)
        print ('Num samples: %i' % num_samples)

        np.random.seed(1)
        np.random.shuffle(self.train_samples)
        num_samples = len(self.train_samples)
        # num_valid_samples = int(num_samples * self.num_valid_samples)
        num_test_samples = int(num_samples * self.test_fraction)
        self.valid_samples = self.train_samples[0:self.num_valid_samples]
        self.test_samples = self.train_samples[self.num_valid_samples:(self.num_valid_samples+num_test_samples)]
        self.train_samples = self.train_samples[(self.num_valid_samples+num_test_samples)::]

        self.environment['num_validation_samples'] = len(self.valid_samples)
        self.environment['num_test_samples'] = len(self.test_samples)







"""
Raw data iterator for the dbpedia/wikipedia dataset
"""
class WikiMultilingualAbstractsProvider(BaseProvider):
    def __init__(self, num_valid_samples=2500, test_fraction=0.05):
        # self.pickle_file_name = 'new_dict_en_da_fr.pkl'
        self.pickle_file_name = 'augmented_multilanguage_dbpedia_full2.pkl'
        self.source_url = 'http://www.intellifind.dk/wiki/%s' % self.pickle_file_name
        self.num_valid_samples = num_valid_samples
        self.test_fraction = test_fraction

    def raw_train_samples_gen(self):
        for target, (categories, texts) in self.train_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def raw_valid_samples_gen(self):
        for target, (categories, texts) in self.valid_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def raw_test_samples_gen(self):
        for target, (categories, texts) in self.test_samples:
            sample = {'class_': target, 'texts': texts, 'categories': categories}
            yield sample

    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        corpus_file = os.path.join(data_dir, self.pickle_file_name)

        if not os.path.exists(data_dir):
            print('creating data directory: %s' % self.environment['config']['data']['data_dir'])
            os.makedirs(self.environment['config']['data']['data_dir'])
        if not os.path.isfile(corpus_file):
            # print('Downloading training data (~124 MB)')
            print('Downloading training data (~1.7 GB)')
            urllib.request.urlretrieve(self.source_url, corpus_file)
            print ('Data download complete')

    def load_data(self):
        filename = os.path.join(self.environment['config']['data']['data_dir'], self.pickle_file_name)
        self.train_samples = pickle.load(open(filename, 'rb'))
        num_samples = 0
        num_samples_de_it_en_fr = 0
        self.num_samples_pr_language = {}
        for _, (_, texts) in self.train_samples.items():
            num_samples += len(texts)
            for l,_ in texts:
                if l in ['en', 'fr', 'de', 'it']:
                    num_samples_de_it_en_fr += 1
                    self.num_samples_pr_language.setdefault(l,0)
                    self.num_samples_pr_language[l] += 1
        print ('Num samples: %i' % num_samples)

        self.train_samples = sorted(self.train_samples.items(), key=lambda x: x[0])
        # counts = {}
        # for sample in self.train_samples:
        #     for l, _ in sample[1][1]:
        #         counts.setdefault(l,0)
        #         counts[l] += 1
        # print ('Samples divided into languages')
        # for lang, c in counts.items():
        #     print('Language %s: %i' % (lang,c))
        # print ('Average number of samples pr topic: %1.4f' % (num_samples/float(len(self.train_samples))))
        # print ('Average number of num_samples_de_it_en_fr pr topic: %1.4f' % (num_samples_de_it_en_fr/float(len(self.train_samples))))
        # self.train_samples = self.train_samples[0:5000]

        np.random.seed(1)
        np.random.shuffle(self.train_samples)
        num_samples = len(self.train_samples)
        # num_valid_samples = int(num_samples * self.num_valid_samples)
        num_test_samples = int(num_samples * self.test_fraction)
        self.valid_samples = self.train_samples[0:self.num_valid_samples]
        self.test_samples = self.train_samples[self.num_valid_samples:(self.num_valid_samples+num_test_samples)]
        self.train_samples = self.train_samples[(self.num_valid_samples+num_test_samples)::]

        self.environment['num_validation_samples'] = len(self.valid_samples)
        self.environment['num_test_samples'] = len(self.test_samples)
        self.environment['num_samples_pr_language'] = self.num_samples_pr_language




"""
Raw data iterator for a list of samples
"""
class SampleListProvider(BaseProvider):
    def __init__(self, samples, valid_frac=0.0, test_frac=0.0):
        idx1 = int(len(samples)*(1.0-valid_frac-test_frac))
        idx2 = int(len(samples)*(1.0-valid_frac))
        self.train_samples = samples[0:idx1]
        self.test_samples = samples[idx1:idx2]
        self.valid_samples = samples[idx2::]

    def raw_train_samples_gen(self):
        for text in self.train_samples:
            sample = {'class_': None, 'text': text}
            yield sample


    def raw_valid_samples_gen(self):
        for text in self.valid_samples:
            sample = {'class_': None, 'text': text}
            yield sample

    def raw_test_samples_gen(self):
        for text in self.test_samples:
            sample = {'class_': None, 'text': text}
            yield sample

    def prepare(self):
        pass

    def load_data(self):
        pass





"""
Raw data iterator for the findzebra dataset
"""
class ExtendedFindzebraProvider(BaseProvider):
    def raw_train_samples_gen(self):
        # Load the raw data as a list of tuples
        print(' - loading data')
        with ZipFile(os.path.join(self.environment['config']['data']['data_dir'], 'findzebra_web_raw.zip'), 'r') as ziparchive:
            for count, filename in enumerate(ziparchive.namelist()[1:]):  # don't include the containing folder as file
                # if count > 5000:
                #     break
                with ziparchive.open(filename, 'r') as f:
                    class_, text = tuple(f.read().decode('utf8').split(' |||| '))
                    sample = {'class_': class_, 'text':text}
                    yield sample

    def raw_valid_samples_gen(self):
        for cui, text in self.valid_samples:
            sample = {'class_': cui.lower(), 'text': text}
            yield sample

    def raw_test_samples_gen(self):
        for cui, text in self.test_samples:
            sample = {'class_': cui, 'text': text}
            yield sample

    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        corpus_file = os.path.join(data_dir, 'findzebra_web_raw.zip')
        validation_queries = os.path.join(data_dir, 'valid_queries.csv')
        test_queries = os.path.join(data_dir, 'test_queries.csv')

        if not os.path.exists(data_dir):
            print('creating data directory: %s' % self.environment['config']['data']['data_dir'])
            os.makedirs(self.environment['config']['data']['data_dir'])
        if not os.path.isfile(corpus_file):
            print('downloading training data (~168 MB)')
            urllib.request.urlretrieve('https://www.dropbox.com/s/085ggy6099y5zfc/findzebra_web_raw.zip?dl=1', corpus_file)
        if not os.path.isfile(validation_queries):
            print('downloading validation data')
            urllib.request.urlretrieve('http://www.intellifind.dk/doctors_dilemma/validation_queries.csv', validation_queries)
        if not os.path.isfile(test_queries):
            print('downloading test queries')
            urllib.request.urlretrieve('http://www.intellifind.dk/doctors_dilemma/test_queries.csv', test_queries)
            print ('Data download complete')

    def load_data(self):
        # with open(os.path.join(self.environment['config']['data']['data_dir'], 'valid_queries.csv'), 'r', encoding='utf-8') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=',')
        #     self.valid_samples = [(row[1], row[0]) for row in reader][1::]
        #     self.environment['num_validation_samples'] = len(self.valid_samples)
        with open(os.path.join(self.environment['config']['data']['data_dir'], 'valid_queries.csv'), 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.valid_samples = [(row[2].lower(), row[1]) for row in reader][1::]
            self.environment['num_validation_samples'] = len(self.valid_samples)

        with open(os.path.join(self.environment['config']['data']['data_dir'], 'test_queries.csv'), 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.test_samples = [(row[2].lower(), row[1]) for row in reader][1::]
            self.environment['num_test_samples'] = len(self.test_samples)


"""
Raw data iterator for the Datasets available from https://arxiv.org/abs/1509.01626

Character-level Convolutional Networks for Text Classification
Xiang Zhang, Junbo Zhao, Yann LeCun
"""

class UniversalArticleDatasetProvider(BaseProvider):
    AG_NEWS = 1
    AMAZON_REVIEW_FULL = 2
    AMAZON_REVIEW_POLARITY = 3
    DBPEDIA = 4
    SOGOU_NEWS = 5
    YAHOO_ANSWERS = 6
    YELP_REVIEW_FULL = 7
    YELP_REVIEW_POLARITY = 8

    def __init__(self, dataset, valid_fraction = 0.05):
        """ The datasets are split into train set and test set.

        Pickle train takes the path to the pickled train set file.
        Pickle test takes the path to the pickled test set file.
        """

        pickle_train = None

        if dataset == self.AG_NEWS:
            pickle_train = 'ag_news_csv_train.pkl'
            pickle_test = 'ag_news_csv_test.pkl'
        if dataset == self.AMAZON_REVIEW_FULL:
            pickle_train = 'amazon_review_full_csv_train.pkl'
            pickle_test = 'amazon_review_full_csv_test.pkl'
        if dataset == self.AMAZON_REVIEW_POLARITY:
            pickle_train = 'amazon_review_polarity_csv_train.pkl'
            pickle_test = 'amazon_review_polarity_csv_test.pkl'
        if dataset == self.DBPEDIA:
            pickle_train = 'dbpedia_csv_train.pkl'
            pickle_test = 'dbpedia_csv_test.pkl'
        if dataset == self.SOGOU_NEWS:
            pickle_train = 'sogou_news_csv_train.pkl'
            pickle_test = 'sogou_news_csv_test.pkl'
        if dataset == self.YAHOO_ANSWERS:
            pickle_train = 'yahoo_answers_csv_train.pkl'
            pickle_test = 'yahoo_answers_csv_test.pkl'
        if dataset == self.YELP_REVIEW_FULL:
            pickle_train = 'yelp_review_full_csv_train.pkl'
            pickle_test = 'yelp_review_full_csv_test.pkl'
        if dataset == self.YELP_REVIEW_POLARITY:
            pickle_train = 'yelp_review_polarity_csv_train.pkl'
            pickle_test = 'yelp_review_polarity_csv_test.pkl'

        assert pickle_train is not None
        assert pickle_test is not None

        self.pickle_file_name_train = pickle_train
        self.source_url_train = 'http://www.intellifind.dk/datasets/%s' % self.pickle_file_name_train
        self.pickle_file_name_test = pickle_test
        self.source_url_test = 'http://www.intellifind.dk/datasets/%s' % self.pickle_file_name_test
        self.valid_fraction = valid_fraction

    def raw_train_samples_gen(self):
        """
        There are some changes to dictionaries depending on the dataset used so be careful!

        For exmaple:

        yahoo_answers_csv_train.pkl: {'answer':str, 'class':int, 'question':str, 'title': str}
        yelp_reviews_full_csv_train.pkl: {'class':int, 'text': str}
        yelp_reviews_polarity_csv_train.pkl: {'class':int, 'text': str}

        All others:

        *_train.pkl: {'class':int, 'text':str, 'title':str}
        """

        for single_dict in self.train_samples:
            yield deepcopy(single_dict)

    def raw_valid_samples_gen(self):
        """
        There are some changes to dictionaries depending on the dataset used so be careful!

        For exmaple:

        yahoo_answers_csv_train.pkl: {'answer':str, 'class':int, 'question':str, 'title': str}
        yelp_reviews_full_csv_train.pkl: {'class':int, 'text': str}
        yelp_reviews_polarity_csv_train.pkl: {'class':int, 'text': str}

        All others:

        *_train.pkl: {'class':int, 'text':str, 'title':str}
        """

        for single_dict in self.valid_samples:
            yield deepcopy(single_dict)

    def raw_test_samples_gen(self):
        """
        There are some changes to dictionaries depending on the dataset used so be careful!

        For exmaple:

        yahoo_answers_csv_test.pkl: {'answer':str, 'class':int, 'question':str, 'title': str}
        yelp_reviews_full_csv_test.pkl: {'class':int, 'text': str}
        yelp_reviews_polarity_csv_test.pkl: {'class':int, 'text': str}

        All others:

        *_test.pkl: {'class':int, 'text':str, 'title':str}
        """

        for single_dict in self.test_samples:
            yield deepcopy(single_dict)



    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        corpus_file_train = os.path.join(data_dir, self.pickle_file_name_train)
        corpus_file_test = os.path.join(data_dir, self.pickle_file_name_test)

        if not os.path.exists(data_dir):
            print('creating data directory: %s' % self.environment['config']['data']['data_dir'])
            os.makedirs(data_dir)
        if not os.path.isfile(corpus_file_train):
            print('Downloading training data (~X MB)')
            urllib.request.urlretrieve(self.source_url_train, corpus_file_train)
            print ('Data dowload complete')
        if not os.path.isfile(corpus_file_test):
            print('Downloading test data (~X MB)')
            urllib.request.urlretrieve(self.source_url_test, corpus_file_test)
            print ('Data download complete')

    def load_data(self):
        filename_train = os.path.join(self.environment['config']['data']['data_dir'],
                                      self.pickle_file_name_train)

        filename_test = os.path.join(self.environment['config']['data']['data_dir'],
                                      self.pickle_file_name_test)

        self.train_samples = pickle.load(open(filename_train, 'rb'))
        self.test_samples = pickle.load(open(filename_test, 'rb'))

        count = 0
        for s in self.train_samples:
            s['class_'] = s['class']
            count += 1
        for s in self.test_samples:
            s['class_'] = s['class']

        print('Num samples: %i' %count)

        np.random.seed(1)
        np.random.shuffle(self.train_samples)
        np.random.shuffle(self.test_samples)

        num_samples = len(self.train_samples)
        num_valid_samples = int(num_samples*self.valid_fraction)

        self.valid_samples = self.train_samples[0:num_valid_samples]

        self.train_samples = self.train_samples[num_valid_samples::]

        self.environment['num_validation_samples'] = len(self.valid_samples)
        self.environment['num_test_samples'] = len(self.test_samples)




"""
Raw data iterator for the pubmed dataset
"""
class PubmedProvider(BaseProvider):
    def raw_train_samples_gen(self):
        for sample_num in self.text_num2text_and_abstract:
            yield self.text_num2text_and_abstract[sample_num]

    def raw_valid_samples_gen(self):
        for cui, text in self.valid_samples:
            sample = {'class_': cui.lower(), 'text': text}
            yield sample

    def raw_test_samples_gen(self):
        for cui, text in self.test_samples:
            sample = {'class_': cui.lower(), 'text': text}
            yield sample

    def prepare(self):
        data_dir = self.environment['config']['data']['data_dir']
        full_text = os.path.join(data_dir, 'text.zip')
        abstracts = os.path.join(data_dir, 'abstracts.zip')
        fz_str2cui = os.path.join(data_dir, 'fz_str2cui.pkl')
        validation_queries = os.path.join(data_dir, 'valid_queries.csv')
        test_queries = os.path.join(data_dir, 'test_queries.csv')
        if not os.path.isdir(self.environment['config']['data']['data_dir']):
            print('creating data directory: %s' % data_dir)
            os.makedirs(data_dir)
        if not os.path.isfile(full_text):
            print('downloading training data (~230 MB)')
            urllib.request.urlretrieve('http://www.intellifind.dk/pubmed/text.zip', full_text)
        if not os.path.isfile(abstracts):
            urllib.request.urlretrieve('http://www.intellifind.dk/pubmed/abstracts.zip',abstracts)
        if not os.path.isfile(fz_str2cui):
            urllib.request.urlretrieve('http://www.intellifind.dk/pubmed/fz_str2cui.pkl',fz_str2cui)

        if not os.path.isfile(validation_queries):
            print('downloading validation data')
            urllib.request.urlretrieve('http://www.intellifind.dk/doctors_dilemma/validation_queries.csv', validation_queries)
        if not os.path.isfile(test_queries):
            print('downloading test queries')
            urllib.request.urlretrieve('http://www.intellifind.dk/doctors_dilemma/test_queries.csv', test_queries)


            print ('Data download complete')

        print(' - loading data')

        fz_str2cui = pickle.load(open(os.path.join(self.environment['config']['data']['data_dir'], 'fz_str2cui.pkl'), 'rb'))
        self.text_num2text_and_abstract = {}
        with ZipFile(os.path.join(self.environment['config']['data']['data_dir'], 'text.zip'), 'r') as ziparchive:
            for count, filename in enumerate(ziparchive.namelist()[1:]):  # don't include the containing folder as file
                with ziparchive.open(filename, 'r') as f:
                    text = f.read().decode('utf8')
                    idx = text.find('\r\n')
                    title = text[0:idx]
                    filename = filename[(filename.find('/')+1)::]
                    ngrams = sum([get_n_grams(nltk.word_tokenize(title.lower()),n) for n in range(1,5)], [])
                    candidate_diseases = [n for n in ngrams if n in fz_str2cui]
                    candidate_disease_cuis = list(set([fz_str2cui[n] for n in candidate_diseases]))
                    self.text_num2text_and_abstract[filename] = {'title':title, 'text': text, 'class_':candidate_disease_cuis}

        with ZipFile(os.path.join(self.environment['config']['data']['data_dir'], 'abstracts.zip'), 'r') as ziparchive:
            for count, filename in enumerate(ziparchive.namelist()[1:]):  # don't include the containing folder as file
                with ziparchive.open(filename, 'r') as f:
                    text = f.read().decode('utf8')
                    idx = text.find('\r\n')
                    title = text[0:idx]
                    filename = filename[(filename.find('/')+1)::]
                    assert title == self.text_num2text_and_abstract[filename]['title']
                    self.text_num2text_and_abstract[filename]['abstract'] = text

        with open(os.path.join(self.environment['config']['data']['data_dir'], 'valid_queries.csv'), 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.valid_samples = [(row[2].lower(), row[1]) for row in reader][1::]
            self.environment['num_validation_samples'] = len(self.valid_samples)

        with open(os.path.join(self.environment['config']['data']['data_dir'], 'test_queries.csv'), 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.test_samples = [(row[2].lower(), row[1]) for row in reader][1::]
            self.environment['num_test_samples'] = len(self.test_samples)


"""
Raw data iterator for the danish findzebra dataset
"""
def html2text(html):
    doc = lh.fromstring(html)
    doc = clean_html(doc)
    doc = doc.text_content()
    doc = re.sub('[\s]{2,}', ' ', doc)
    return doc


class DanishCorpusProvider(BaseProvider):
    def __init__(self):
        pass

    def raw_train_samples_gen(self):
        # Load the raw data as a list of tuples
        print(' - loading data')
        with open(os.path.join(self.environment['config']['data']['data_dir'], 'danish_corpus.csv'), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['text'] = html2text(row['content'])
                yield row

    def raw_valid_samples_gen(self):
        return []

    def prepare(self):
        if not os.path.exists(self.environment['config']['data']['data_dir']):
            import urllib.request
            print('creating data directory: %s' % self.environment['config']['data']['data_dir'])
            os.makedirs(self.environment['config']['data']['data_dir'])
            print('downloading training data (~18 MB)')
            urllib.request.urlretrieve('http://www.bioware.dk/danish_corpus/danish_corpus.csv',
                                       os.path.join(self.environment['config']['data']['data_dir'], 'danish_corpus.csv'))


            urllib.request.urlretrieve('http://www.bioware.dk/danish_corpus/cui2danish_names.pkl',
                                       os.path.join(self.environment['config']['data']['data_dir'], 'cui2danish_names.pkl'))



            urllib.request.urlretrieve('http://www.bioware.dk/danish_corpus/cui2prefered_danish_name.pkl',
                                       os.path.join(self.environment['config']['data']['data_dir'], 'cui2prefered_danish_name.pkl'))



    def load_data(self):
        self.environment['num_validation_samples'] = 0


if __name__ == '__main__':
    from flow_framework import *
    flow = [ConfigLoaderFlowComponent('fz search example/bording_config.ini'),
            BordingProvider()]
    env = FlowController(flow).execute()
    for (cui, text) in env['raw_test_samples_gen']():
        print ('%s: %i' % (cui, len(text)))
