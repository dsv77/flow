from keras import backend as K

def top_k_accuracy(k):
    def top_k(y_true, y_pred):
        hits = K.in_top_k(K.cast(y_pred, 'float32'), K.cast(K.squeeze(y_true,1), 'int32'), k)
        return K.mean(hits)
    top_k.__name__ = 'top {}'.format(k)
    return top_k

