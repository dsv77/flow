import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping
import os
import keras.backend as K
from flow_framework import AbstractFlowComponent
import os

def get_accuracy(model, generator, k_list = [1, 5, 10, 20]):
    x_samples, y_samples = next(generator)
    x_probs = []
    y_vector = []
    # accuracies = []
    for sample_num in range(0, len(x_samples)):
        if sample_num > 0 and (sample_num% 500 == 0):
            print ('Finished %i samples: ' % sample_num)
        x = x_samples[sample_num]
        probs = np.mean(model.predict(x), axis=0)
        x_probs.append(probs)
        y_vector.append(y_samples[sample_num][0])
    y_vector = np.asarray(y_vector, 'int32')
    y_pred = np.vstack(x_probs).astype('float32')


    accuracies = []
    for sample_num in k_list:
        highest_probs_idx = np.argsort(y_pred, axis=1)[:, -sample_num:]
        y_true_mat = np.repeat(y_vector.reshape((-1, 1)), sample_num, axis=1)
        is_hit = np.equal(highest_probs_idx, y_true_mat)
        accuracy = np.mean(np.any(is_hit, axis=1))
        accuracies.append(accuracy)
    return accuracies


class PrintTopWordsCallback(Callback):
    def __init__(self, valid_batch_generator, token2id, num_buckets, get_probabilities_function=None, **kwargs):
        super(PrintTopWordsCallback, self).__init__(**kwargs)
        self.get_probabilities_function = get_probabilities_function
        self.valid_batch_generator = valid_batch_generator
        self.current_best_accuracy = 0.0
        self.id2token = dict((a, b) for (b, a) in token2id.items())
        self.token2id = token2id
        self.bucket2tokens = {}
        for token, id in token2id.items():
            if id > 0:
                h = (id%num_buckets)+1
                self.bucket2tokens.setdefault(h, [])
                self.bucket2tokens[h].append(token)
        self.bucket2tokens[0] = 'ZERO_TOKEN'

        self.id2token[0] = 'ZERO_TOKEN'


    def on_epoch_begin(self, epoch, logs={}):
        try:
            p = self.get_probabilities_function()
            print(np.sum(np.abs(p)))

            # for bucket_num, tokens in self.bucket2tokens.items():
            #     token_ids = [self.token2id[token] for token in tokens]
            #     bucket_
            # abs_p = np.max(np.abs(p), 1**(-8))
            # sum_probs = np.sum(abs_p, axis=1)
            # probs = abs_p/np.expand_dims(sum_probs, 1)
            # entropy = np.sum(probs*np.log(probs), 1)
            # print(entropy[0:4,:])


            idx = np.argsort(np.max(np.abs(p), axis=1)).tolist()

            bottom_words = [self.id2token[i] for i in idx[0:20]]
            top_words= [self.id2token[i] for i in idx[-20::]]
            print ("top words:\t\t %s" % ', '.join(top_words))
            print ("Bottom words:\t %s" % ', '.join(bottom_words))
        except Exception as e:
            print("Unable to print top words"+str(e))
        print('')





class AddTensorboardLoggingCallbackFlowComponent(AbstractFlowComponent):
    def __init__(self, name):
        super(self.__class__, self).__init__()
        self.name = name

    def execute(self, environment={}):
        print('Adding tensorbord logger...')
        if 'callbacks' not in environment:
            environment['callbacks'] = []
        all_callbacks = environment['callbacks']

        if K.backend() == 'tensorflow':
            log_path = './logs/{}'.format(self.name)
            all_callbacks.append(TensorBoard(log_dir=log_path, write_graph=False))

        environment['callbacks'] = all_callbacks

        return environment


class AddPrintTopWordsCallbackFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()

    def execute(self, environment={}):
        if 'multi_hashing_layer' not in environment:
            return environment
        if 'callbacks' not in environment:
            environment['callbacks'] = []
        all_callbacks = environment['callbacks']

        valid_batch_generator = environment['encoded_valid_samples_gen']

        try:
            probs = environment['multi_hashing_layer'].get_probabilities
        except:
            probs = None
        num_buckets = environment['multi_hashing_layer'].num_buckets
        d = PrintTopWordsCallback(valid_batch_generator=valid_batch_generator,
                                  token2id=environment['token2id'], num_buckets=num_buckets, get_probabilities_function=probs)
        all_callbacks.append(d)
        environment['callbacks'] = all_callbacks
        print('Show top wordsFlowComponent added')
        return environment

class ShowPredictionsCallvack(Callback):
    def __init__(self, sentences, transform_fun, class2id, token2id, cui2prefered_name, num_predictions, **kwargs):
        super(ShowPredictionsCallvack, self).__init__(**kwargs)
        self.sentences = sentences
        self.class2id = class2id
        self.transform_fun = transform_fun
        self.id2class = dict([(i, c) for (c, i) in self.class2id.items()])
        self.token2id = token2id
        self.cui2prefered_name = cui2prefered_name
        self.num_predictions = num_predictions

    def on_epoch_begin(self, epoch, logs={}):
        for s in self.sentences:
            encoded_str = self.transform_fun(s)
            predictions = self.model.predict(encoded_str)
            predictions = predictions[0, :].flatten()
            top_ids = np.argsort(-predictions)[0:self.num_predictions]
            suggestions = [self.id2class[i] for i in top_ids]
            probs = [predictions[i] for i in top_ids.tolist()[0:self.num_predictions]]
            suggestions = [(self.cui2prefered_name[cui], p)
                           for (cui, p) in zip(suggestions,probs) if cui in self.cui2prefered_name and p > 0.01]
            print ("%s --> %s" % (s, suggestions))
            print('----------------')


class AddShowPredictionsCallbackFlowComponent(AbstractFlowComponent):
    def __init__(self, sentences):
        super(self.__class__, self).__init__()
        self.sentences = sentences

    def execute(self, environment={}):
        if 'callbacks' not in environment:
            environment['callbacks'] = []
        all_callbacks = environment['callbacks']

        self.transformations = environment['transformations']
        self.transform_functions = ([t[0] for t in self.transformations])

        def encode(text):
            for t in self.transform_functions:
                text = t(text, environment)
            return text
        self.transform_fun = encode

        class2id = environment['class2id']
        token2id = environment['token2id']
        cui2prefered_name = environment['cui2prefered_name']
        d = ShowPredictionsCallvack(self.sentences, self.transform_fun, class2id, token2id, cui2prefered_name, 10)
        all_callbacks.append(d)
        environment['callbacks'] = all_callbacks
        print('Show predictions callback added')
        return environment


class LogProgressCallback(Callback):
    def __init__(self, log_dir, out_file_name, overwrite=True, **kwargs):
        super(LogProgressCallback, self).__init__(**kwargs)
        self.out_file_name = out_file_name
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        file_name = os.path.join(self.log_dir, self.out_file_name)
        if os.path.isfile(file_name) and overwrite:
            open(file_name, 'a').close()


    def on_epoch_end(self, epoch, logs={}):
        import csv
        filename =os.path.join(self.log_dir, self.out_file_name)
        rows = [logs]
        if os.path.isfile(filename):
            reader = csv.DictReader(open(filename, 'r', encoding='utf-8'), lineterminator='\n', delimiter='\t')
            rows = [r for r in reader] + [logs]
        writer = csv.DictWriter(open(os.path.join(self.log_dir, self.out_file_name),'w',encoding='utf-8'),
                                fieldnames=list(logs.keys()), lineterminator='\n', delimiter='\t')
        writer.writeheader()
        try:
            writer.writerows(rows)
        except Exception as e:
            print('Unable to write to log: ' + str(e))

class AddLogProgressCallbackFlowComponent(AbstractFlowComponent):
    def __init__(self, out_dir, out_file):
        super(self.__class__, self).__init__()
        self.out_dir = out_dir
        self.out_file = out_file

    def execute(self, environment={}):
        print('Adding model logging callback...')
        if 'callbacks' not in environment:
            environment['callbacks'] = []
        all_callbacks = environment['callbacks']
        all_callbacks.append(LogProgressCallback(self.out_dir, self.out_file))
        return environment


class ClearCallbacksFlowComponent(AbstractFlowComponent):
    def __init__(self):
        super(self.__class__, self).__init__()

    def execute(self, environment={}):
        print('Clearing callbacks...')
        environment['callbacks'] = []
        return environment


class AddEarlyStoppingCallbackFlowComponent(AbstractFlowComponent):
    def __init__(self, metric='val_loss', patience=0):
        super(self.__class__, self).__init__()
        self.metric = metric
        self.patience = patience

    def execute(self, environment={}):
        if environment['num_validation_samples']==0:
            return environment
        print('Adding early stopping callback...')
        if 'callbacks' not in environment:
            environment['callbacks'] = []
        all_callbacks = environment['callbacks']
        all_callbacks.append(EarlyStopping(monitor=self.metric, patience=self.patience))
        return environment



class AddModelSaveCallbackFlowComponent(AbstractFlowComponent):
    def __init__(self, checkpoint_path, monitor='val_loss', save_best_only=True):
        super(self.__class__, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.monitor = monitor
        self.save_best_only = save_best_only

    def execute(self, environment={}):
        print('Adding model save callback...')
        if 'callbacks' not in environment:
            environment['callbacks'] = []
        all_callbacks = environment['callbacks']

        # monitor = 'val_loss'
        # if environment['num_validation_samples'] == 0:
        #     monitor = 'loss'
        #

        checkpoint_name = '/weights.{%s:011.8f}-{epoch:06d}.hdf5' % self.monitor
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        all_callbacks.append(ModelCheckpoint(self.checkpoint_path + checkpoint_name,
                                             save_weights_only=True,
                                             save_best_only=self.save_best_only, monitor=self.monitor))



        environment['callbacks'] = all_callbacks
        return environment

"""
AddCheckPointsCleanCallbackFlowComponent removes all but the best checkpoint file
"""
class AddCheckpointsCleanCallbackFlowComponent(AbstractFlowComponent):
    def __init__(self, checkpoint_path, order='ascending'):
        super(self.__class__, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.order = order

    def execute(self, environment={}):
        print('Adding checkpoint cleaning callback...')
        if 'callbacks' not in environment:
            environment['callbacks'] = []
        all_callbacks = environment['callbacks']
        all_callbacks.append(CleanCheckpointsCallback(self.checkpoint_path, self.order))
        environment['callbacks'] = all_callbacks
        return environment





class CleanCheckpointsCallback(Callback):
    def __init__(self, checkpoints_path, order, **kwargs):
        super(self.__class__, self).__init__()
        self.checkpoints_path = checkpoints_path
        self.order = order


    def on_epoch_end(self, epoch, logs={}):
        if os.path.exists(self.checkpoints_path):
            checkpoint_files = os.listdir(self.checkpoints_path)
            if checkpoint_files:
                if self.order == 'ascending':
                    checkpoint_to_remove = sorted(os.listdir(self.checkpoints_path))[1::]
                else:
                    checkpoint_to_remove = sorted(os.listdir(self.checkpoints_path))[0:-1]
                for file in checkpoint_to_remove:
                    os.remove(os.path.join(self.checkpoints_path, file))