from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import merge, Dense, TimeDistributed, Merge
import numpy as np
from keras.regularizers import l2
from Crypto.Util import number
# import theano

class Attention(Layer):
    def __init__(self, inner_module=None, **kwargs):
        """Attention mechanism over the time steps of the input.

        The importance of each time step from the input will be
        calculated by using softmax on the outputs of the inner_module
        applied to each time step of the input.

        The inner_module can be any model or layer that can applied to
        the individual time steps of the input and has a single scalar as
        its output for each.

        Note that when you define your own inner_module it is your own
        responsibility to make sure that the input and output dimensions
        are correct. 
        
        Args:
            inner_module: (Default: None) Defines the inner_module to use
                for the attention mechanism. There are 2 possible ways to
                define it: 1. If set to None, a simple Dense layer will be
                used. 2. If a Keras model is given, that model will be
                used directly.
        """
        self.supports_masking = True
        self.inner_module = inner_module

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_input_features = input_shape[2]
        self.num_time_steps = input_shape[1]

        if not self.inner_module:
            self.inner_module = Sequential([Dense(1,
                                            input_shape=[self.num_input_features])])

        # TODO: saving inner_module weights with checkpoints

    def call(self, x, mask=None):
        mask = K.cast(mask, K.floatx())

        # Apply inner_module to each time step
        x_transformed = TimeDistributed(self.inner_module)(x)
        x_trans_reshaped = K.reshape(x_transformed, [-1, self.num_time_steps])

        # Compute Alpha
        x_trans_masked = merge([x_trans_reshaped, mask], 'mul')
        alpha = K.softmax(x_trans_masked)
        alpha_reshaped = K.reshape(alpha, [-1, 1, self.num_time_steps])

        # Compute the weighted average of the input time steps (using alpha as weights)
        output = K.batch_dot(alpha_reshaped, x)
        return K.batch_flatten(output)
    
    def compute_mask(self, input, mask):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

# class WordMasking(Layer):
#     def __init__(self, space_indices, **kwargs):
#         self.supports_masking = True
#         self.space_indices = space_indices
#         super(WordMasking, self).__init__(**kwargs)
#
#     def call(self, input, mask=None):
#         s = K.cast(K.equal(input, self.space_indices[0]), 'int32')
#         for m in self.space_indices[1::]:
#             s += K.cast(K.equal(input, m), 'int32')
#         return s
#
#     def get_output_shape_for(self, input_shape):
#         return (2,) + input_shape



class ReduceMean(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True

        super(ReduceMean, self).__init__(**kwargs)

    def call(self, x, mask=None):
        x, m = x
        x = x * K.cast(K.expand_dims(K.not_equal(m, 0), -1), 'float32')
        x = K.cast(x, 'float32')
        return K.mean(x, axis=1, keepdims=False)

    def compute_mask(self, input, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])



class WordExtraction(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WordExtraction, self).__init__(**kwargs)


    def call(self, input_vars, mask=None):
        m = K.cast(input_vars[1] * K.cast(K.greater_equal(input_vars[1], 0), 'int32'), 'int32')
        def step_function(x_s, ind):
            v = K.gather(x_s, ind.flatten())
            return v
        results, _ = theano.scan(step_function, sequences=[input_vars[0], m])
        # deal with Theano API inconsistency
        if type(results) is list:
            outputs = results[0]
        else:
            outputs = results
        return outputs

    def compute_mask(self, input_vars, mask=None):
        m = K.greater_equal(input_vars[1], 0)
        return m

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], None, input_shape[0][2])

class WordExtractionBidirectional(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WordExtractionBidirectional, self).__init__(**kwargs)

    def call(self, input_vars, mask=None):
        m_fwd = K.cast(input_vars[2] * K.cast(K.greater_equal(input_vars[2], 2), 'int32'), 'int32') # forward mask
        m_back = K.cast(input_vars[3] * K.cast(K.greater_equal(input_vars[3], 2), 'int32'), 'int32')  # backward mask

        def step_function(x_forward, x_backward, indices_fwd, indices_back):
            v1 = K.gather(x_forward, indices_fwd.flatten())
            v2 = K.gather(x_backward, indices_back.flatten())
            v = K.concatenate([v1,v2], axis=-1)
            return v

        results, _ = theano.scan(step_function, sequences=[input_vars[0], input_vars[1], m_fwd, m_back])
        # deal with Theano API inconsistency
        if type(results) is list:
            outputs = results[0]
        else:
            outputs = results
        return outputs

    def compute_mask(self, input_vars, mask=None):
        m = K.greater_equal(input_vars[2], 0)
        return m

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], None, input_shape[0][2]*2)

# class WordMasking(Layer):
#     def __init__(self, space_indices, embedding_size, **kwargs):
#         self.supports_masking = True
#         self.space_indices = space_indices
#         self.embedding_size = embedding_size
#         super(WordMasking, self).__init__(**kwargs)
#
#     def call(self, input, mask=None):
#         s = sum([K.cast(K.equal(input, m), 'int32') for m in self.space_indices])
#         return K.repeat_elements(s,self.embedding_size,-1)
#
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape +(self.embedding_size,)



class TemporalDropOut(Layer):
    def __init__(self, p, **kwargs):
        self.supports_masking = True
        self.dropout = p
        super(TemporalDropOut, self).__init__(**kwargs)

    def call(self, x, mask=None):
        retain_p = 1. - self.dropout
        rand_binom = K.random_binomial((x.shape[0],x.shape[1]), p=retain_p) * (1. / retain_p)
        rand_binom = K.expand_dims(rand_binom,-1)
        return K.in_train_phase((rand_binom * x)/retain_p, x)


    def compute_mask(self, input, mask):
        return mask

    def get_output_shape_for(self, input_shape):
        return (input_shape)


class ReduceSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True

        super(ReduceSum, self).__init__(**kwargs)

    def call(self, x, mask=None):
        x, m = x
        x = x * K.cast(K.expand_dims(K.not_equal(m,0), -1), 'float32')
        x = K.cast(x, 'float32')
        return K.sum(x, axis=1,keepdims=False)

    def compute_mask(self, input, mask):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    # def get_output_shape_for(self, input_shape):
    #     return (input_shape[0], input_shape[2])

class WordImportanceLayer(Layer):
    def __init__(self, inner_module=None, **kwargs):
        """Attention mechanism over the time steps of the input.

        The importance of each time step from the input will be
        calculated by using softmax on the outputs of the inner_module
        applied to each time step of the input.

        The inner_module can be any model or layer that can applied to
        the individual time steps of the input and has a single scalar as
        its output for each.

        Note that when you define your own inner_module it is your own
        responsibility to make sure that the input and output dimensions
        are correct.
        """
        self.supports_masking = True
        self.trainable_weights = self.inner_module.trainable_weights

        super(WordImportanceLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_input_features = input_shape[1]
        self.weight = K.variable(np.random.uniform(0,1,(1, self.num_input_features)))


    def call(self, x, mask=None):
        mask = K.cast(mask, K.floatx())

        # Apply inner_module to each time step
        x_transformed = TimeDistributed(self.inner_module)(x)
        x_trans_reshaped = K.reshape(x_transformed, [-1, self.num_time_steps])

        # Compute Alpha
        x_trans_masked = merge([x_trans_reshaped, mask], 'mul')
        alpha = K.softmax(x_trans_reshaped)
        alpha_reshaped = K.reshape(alpha, [-1, 1, self.num_time_steps])

        # Compute the weighted average of the input time steps (using alpha as weights)
        output = K.batch_dot(alpha_reshaped, x)
        return K.batch_flatten(output)

    def compute_mask(self, input, mask):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])


class HashingLayerDropout(Layer):
    def __init__(self, num_buckets, feature_offset, word_count_including_zero_token, p_init = 1.0, sum_buckets = True, W=None, W_trainable=False, reg_factor=0.001, mask_zero=False, append_weight= False, **kwargs):
        super(HashingLayerDropout, self).__init__(**kwargs)
        self.word_count = word_count_including_zero_token
        self.num_buckets = num_buckets
        self.feature_offset = feature_offset
        self.mask_zero = mask_zero
        self.append_weight = append_weight
        self.p = None
        self.trainable_weights = []
        if p_init is not None:
            p = (np.ones((word_count_including_zero_token,), dtype='float32') * p_init).astype('float32')
            self.p = K.variable(p,name='p_hash')
            self.trainable_weights.append(self.p)
            self.get_probs = K.function([], [self.p])
        if W is None:
            # one hot encoding
            W = np.row_stack((np.zeros((1, num_buckets),dtype='float32'), np.eye(num_buckets,dtype='float32')))
        else:
            W = np.row_stack((np.zeros((1, W.shape[1])), W)).astype('float32')
        self.embedding_size = W.shape[1]
        W_shared = K.variable(W, name='W_hash')
        self.W = W_shared
        self.sum_buckets = sum_buckets
        if W_trainable:
            self.trainable_weights.append(self.W)

        if self.p is not None and reg_factor > 0:
            reg = l2(reg_factor)
            reg.set_param(self.p)
            self.regularizers.append(reg)

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def call(self, input, mask=None):
        W = self.W
        idx_bucket = ((input+self.feature_offset)%self.num_buckets)
        idx_bucket = (idx_bucket +1)* (1-K.cast(K.equal(0, input), 'int32'))
        W0 = K.gather(W, idx_bucket)
        if self.p is not None:
            p_0 = K.gather(self.p, input)
            p = K.expand_dims(p_0,dim=-1)
            retval = W0*p
            if self.append_weight:
                retval = K.concatenate([retval,p],axis=-1)
        else:
            retval = W0
        if self.sum_buckets:
            return K.max(retval, axis=1)
        else:
            return retval


    def get_output_shape_for(self, input_shape):
        weight_addition = 0
        if self.append_weight:
            weight_addition = 1
        if self.sum_buckets:
            return (input_shape[0], self.embedding_size+weight_addition)
        else:
            return (input_shape[0], input_shape[1], self.embedding_size+weight_addition)


    def get_probabilities(self):
        if self.p is not None:
            if K.backend() == 'theano':
                return self.p.get_value()
            else:
                return self.get_probs([]) #self.p
        else:
            return None

def print_matrix(m, n):
    for i in range(m.shape[0]):
        print('\t'.join([('%1.'+str(n)+'f') % (e) for e in m[i, :].tolist()]))


class MultiHashingLayerDropout(Layer):
    def __init__(self, W, word_count_including_zero_token, p_init = None,
                 W_trainable=True, p_trainable = True, reg_factor=0.00001,
                 mask_zero=False, append_weight= False, aggregation_mode = 'sum', **kwargs):
        super(MultiHashingLayerDropout, self).__init__(**kwargs)
        np.random.seed(3)
        self.word_count = word_count_including_zero_token
        self.num_buckets = W.shape[0]
        self.mask_zero = mask_zero
        self.append_weight = append_weight
        self.p = None
        self.trainable_weights = []
        self.p_trainable = p_trainable

        if p_init is None:
            self.num_hash_funs = 1
            p_init = np.ones((self.word_count, self.num_hash_funs))
            self.p_trainable = False
        assert (len(p_init.shape) == 2)
        self.num_hash_funs = p_init.shape[1]
        self.hashing_vals = []
        self.hashing_offset_vals = []

        tab = (np.random.randint(0, 2**16,size=(self.word_count, self.num_hash_funs)) % self.num_buckets)+1
        self.hash_tables = K.variable(tab, dtype='int32')

        # print_matrix(tab,0)
        self.p = K.variable(p_init,name='p_hash')

        if self.p_trainable:
            self.trainable_weights.append(self.p)
        self.get_probs = K.function([], [self.p])

        # add zero vector for nulls (for masking)
        W = np.row_stack((np.zeros((1, W.shape[1])), W)).astype('float32')

        self.embedding_size = W.shape[1]
        W_shared = K.variable(W, name='W_hash')
        self.W = W_shared
        if W_trainable:
            self.trainable_weights.append(self.W)

        if self.p is not None and reg_factor > 0:
            reg = l2(reg_factor)
            reg.set_param(self.p)
            self.regularizers.append(reg)
        if aggregation_mode == 'sum':
            self.aggregation_function = sum
        else:
            if aggregation_mode == 'concatenate':
                self.aggregation_function = lambda x: K.concatenate(x,axis = -1)
            else:
                raise('unknown aggregation function')
        self.aggregation_mode = aggregation_mode

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def call(self, input, mask=None):
        W = self.W
        pvals = []
        retvals = []
        idx_bucket_all = K.gather(self.hash_tables, input)
        for hash_fun_num in range(self.num_hash_funs):

            # idx_bucket = ((input+self.hashing_vals[hash_fun_num])%self.num_buckets)
            # idx_bucket = (idx_bucket +1)* (1-K.cast(K.equal(0, input), 'int32'))
            W0 = K.gather(W, idx_bucket_all[:,:,hash_fun_num]*(1-K.cast(K.equal(0, input), 'int32')))
            p_0 = K.gather(self.p[:,hash_fun_num], input)
            p = K.expand_dims(p_0, -1)
            pvals.append(p)
            retvals.append(W0*p)
        retval = self.aggregation_function(retvals)
        if self.append_weight:
            retval = K.concatenate([retval]+pvals,axis=-1)
        return retval


    def get_output_shape_for(self, input_shape):
        weight_addition = 0
        if self.append_weight:
            weight_addition = self.num_hash_funs
        if self.aggregation_mode == 'sum':
            return (input_shape[0], input_shape[1], self.embedding_size+weight_addition)
        else:
            return (input_shape[0], input_shape[1], self.embedding_size*self.num_hash_funs + weight_addition)


    def get_probabilities(self):
        if self.p is not None:
            if K.backend() == 'theano':
                return self.p.get_value()
            else:
                return self.get_probs([])[0] #self.p
        else:
            return None




class HashEmbeddingMod(Layer):
    def __init__(self, W, word_count_including_zero_token, p_init = None,
                 W_trainable=True, p_trainable = True, reg_factor=0.00001,
                 mask_zero=False, append_weight= False, aggregation_mode = 'sum', seed=3, **kwargs):
        super(HashEmbeddingMod, self).__init__(**kwargs)
        np.random.seed(seed)
        self.word_count = word_count_including_zero_token
        print (self.word_count)
        self.num_buckets = W.shape[0]
        self.mask_zero = mask_zero
        self.append_weight = append_weight
        self.p = None
        self.trainable_weights = []
        self.p_trainable = p_trainable

        if p_init is None:
            self.num_hash_funs = 1
            p_init = np.ones((self.word_count, self.num_hash_funs))
            self.p_trainable = False
        assert (len(p_init.shape) == 2)
        self.num_hash_funs = p_init.shape[1]
        self.hashing_vals = []
        self.hashing_offset_vals = []


        self.hash_primes = [int(number.getPrime(16)) for i in range(self.num_hash_funs)]
        print (self.hash_primes)

        # tab = (np.random.randint(0, 2**30,size=(self.word_count, self.num_hash_funs)) % self.num_buckets)+1
        # self.hash_tables = K.variable(tab, dtype='int32')

        # print_matrix(tab,0)
        self.p = K.variable(p_init,name='p_hash')

        if self.p_trainable:
            self.trainable_weights.append(self.p)
        self.get_probs = K.function([], [self.p])

        # add zero vector for nulls (for masking)
        W = np.row_stack((np.zeros((1, W.shape[1])), W)).astype('float32')

        self.embedding_size = W.shape[1]
        W_shared = K.variable(W, name='W_hash')
        self.W = W_shared
        if W_trainable:
            self.trainable_weights.append(self.W)

        if self.p is not None and reg_factor > 0:
            reg = l2(reg_factor)
            reg.set_param(self.p)
            self.regularizers.append(reg)
            # reg = l2(reg_factor)
            # reg.set_param(self.W)
            # self.regularizers.append(reg)
        if aggregation_mode == 'sum':
            self.aggregation_function = sum
        else:
            if aggregation_mode == 'concatenate':
                self.aggregation_function = lambda x: K.concatenate(x,axis = -1)
            else:
                raise('unknown aggregation function')
        self.aggregation_mode = aggregation_mode

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def call(self, input, mask=None):
        W = self.W
        pvals = []
        retvals = []
        input_w = input%self.word_count
        input_p = (3+input)%self.word_count
        # idx_bucket_all = K.gather(self.hash_tables, input_w)
        for hash_fun_num in range(self.num_hash_funs):
            # idx_bucket =
            # idx_bucket =
            # idx_bucket =
            W0 = K.gather(W, K.cast(K.cast(input_w,'int32') % self.hash_primes[hash_fun_num] % self.num_buckets, 'int32')*(1-K.cast(K.equal(0, input_w), 'int32')))
            idx_bucket = 0
            p_0 = K.gather(self.p[:,hash_fun_num], input_p)
            p = K.expand_dims(p_0,dim=-1)
            pvals.append(p)
            retvals.append(W0*p)
        retval = self.aggregation_function(retvals)
        if self.append_weight:
            retval = K.concatenate([retval]+pvals,axis=-1)
        return retval


    def get_output_shape_for(self, input_shape):
        weight_addition = 0
        if self.append_weight:
            weight_addition = self.num_hash_funs
        if self.aggregation_mode == 'sum':
            return (input_shape[0], input_shape[1], self.embedding_size+weight_addition)
        else:
            return (input_shape[0], input_shape[1], self.embedding_size*self.num_hash_funs + weight_addition)


    def get_probabilities(self):
        if self.p is not None:
            if K.backend() == 'theano':
                return self.p.get_value()
            else:
                return self.get_probs([])[0] #self.p
        else:
            return None




class HashEmbedding(Layer):
    def __init__(self, W, word_count_including_zero_token, p_init = None,
                 W_trainable=True, p_trainable = True, reg_factor=0.00001,
                 mask_zero=False, append_weight= False, aggregation_mode = 'sum', seed=3, **kwargs):
        super(HashEmbedding, self).__init__(**kwargs)
        np.random.seed(seed)
        self.word_count = word_count_including_zero_token
        print (self.word_count)
        self.num_buckets = W.shape[0]
        self.mask_zero = mask_zero
        self.append_weight = append_weight
        self.p = None
        self.trainable_weights = []
        self.p_trainable = p_trainable

        if p_init is None:
            self.num_hash_funs = 1
            p_init = np.ones((self.word_count, self.num_hash_funs))
            self.p_trainable = False
        assert (len(p_init.shape) == 2)
        self.num_hash_funs = p_init.shape[1]
        self.hashing_vals = []
        self.hashing_offset_vals = []

        tab = (np.random.randint(0, 2**30,size=(self.word_count, self.num_hash_funs)) % self.num_buckets)+1
        self.hash_tables = K.variable(tab, dtype='int32')

        # print_matrix(tab,0)
        self.p = K.variable(p_init,name='p_hash')

        if self.p_trainable:
            self.trainable_weights.append(self.p)
        self.get_probs = K.function([], [self.p])

        # add zero vector for nulls (for masking)
        W = np.row_stack((np.zeros((1, W.shape[1])), W)).astype('float32')

        self.embedding_size = W.shape[1]
        W_shared = K.variable(W, name='W_hash')
        self.W = W_shared
        if W_trainable:
            self.trainable_weights.append(self.W)

        if self.p is not None and reg_factor > 0:
            reg = l2(reg_factor)
            reg.set_param(self.p)
            self.regularizers.append(reg)
            # reg = l2(reg_factor)
            # reg.set_param(self.W)
            # self.regularizers.append(reg)
        if aggregation_mode == 'sum':
            self.aggregation_function = sum
        else:
            if aggregation_mode == 'concatenate':
                self.aggregation_function = lambda x: K.concatenate(x,axis = -1)
            else:
                raise('unknown aggregation function')
        self.aggregation_mode = aggregation_mode

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def call(self, input, mask=None):
        W = self.W
        pvals = []
        retvals = []
        input_w = input%self.word_count
        input_p = (3+input)%self.word_count
        idx_bucket_all = K.gather(self.hash_tables, input_w)
        for hash_fun_num in range(self.num_hash_funs):
            W0 = K.gather(W, idx_bucket_all[:,:,hash_fun_num]*(1-K.cast(K.equal(0, input_w), 'int32')))
            p_0 = K.gather(self.p[:,hash_fun_num], input_p)
            p = K.expand_dims(p_0,dim=-1)
            pvals.append(p)
            retvals.append(W0*p)
        retval = self.aggregation_function(retvals)
        if self.append_weight:
            retval = K.concatenate([retval]+pvals,axis=-1)
        return retval


    def get_output_shape_for(self, input_shape):
        weight_addition = 0
        if self.append_weight:
            weight_addition = self.num_hash_funs
        if self.aggregation_mode == 'sum':
            return (input_shape[0], input_shape[1], self.embedding_size+weight_addition)
        else:
            return (input_shape[0], input_shape[1], self.embedding_size*self.num_hash_funs + weight_addition)


    def get_probabilities(self):
        if self.p is not None:
            if K.backend() == 'theano':
                return self.p.get_value()
            else:
                return self.get_probs([])[0] #self.p
        else:
            return None


def print_matrix(m, n):
    for i in range(m.shape[0]):
        print('\t'.join([('%1.'+str(n)+'f') % (e) for e in m[i, :].tolist()]))

if __name__ == '__main__':
    num_features = 4
    offset = 0
    num_words = 10
    words_in_sent = 3
    num_hash_funs = 2
    embedding_size = 5
    import random
    np.random.seed(2)


    encoded_text_test = np.random.randint(0,num_words,size=(2,words_in_sent)).astype('int32')



    print (encoded_text_test)
    from keras.layers import Input
    l_in_answer = Input([words_in_sent], dtype='int32')

    # sess.run(l_in_answer, feed_dict={l_in_answer: encoded_text_test})

    p_init = np.random.normal(0, 1, size=(num_words, num_hash_funs))
    # w_init = np.random.normal(0, 1, size=(num_features, embedding_size))
    w_init = np.asarray(range(0, num_features*embedding_size)).reshape(num_features, embedding_size)
    print('W')
    print (w_init)
    print ()

    print('p_init')
    print_matrix(p_init, 3)
    print ()
    # num_features = 15000
    aggregation_mode = 'sum'
    # aggregation_mode = 'concatenate'
    m = MultiHashingLayerDropout(W=w_init, word_count_including_zero_token=num_words, p_init=p_init, aggregation_mode=aggregation_mode,
                                 W_trainable=True, p_trainable=True, append_weight=True, name='l_cast_emb_input')
    l_cast_emb_input = m(l_in_answer)

    f = K.function([l_in_answer],[l_cast_emb_input])
    # print(f([encoded_text_test])[0].shape)
    # idx = (encoded_text_test % num_features)+1
    # for i, v in enumerate(list(idx)):
    #     print(v)
    np.set_printoptions(formatter={'all': lambda x: '%1.3f' % x})
    u = f([encoded_text_test])[0]
    print (u.shape)
    u = u[0,:,:]
    # print (np.asarray(u[0,:,:]))
    print ('U')
    print_matrix(u, 3)
    import pprint
    # pprint.pprint(u[0,:,:])
    # print ("l_cast_emb_input: %s" % str(f([encoded_text_test])))
    # print (m.get_probabilities())
    print()
