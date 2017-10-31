import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
from sklearn.manifold.t_sne import TSNE
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics.scorer import make_scorer
from sklearn.metrics.pairwise import cosine_similarity

def fit_tsnee(vectors, learning_rates=[1000], metric='euclidean'):
    vectors = np.vstack(vectors)
    if vectors.shape[1] == 2:
        return vectors, None, None

    if vectors.shape[1] > 250:
        from sklearn.decomposition import PCA
        modelPCA = PCA(n_components=250)
        np.set_printoptions(suppress=False)
        print('Fitting PCA...')
        X_hat = modelPCA.fit_transform(vectors)
        print('Done fitting PCA. Explained variance %1.4f' % np.sum(modelPCA.explained_variance_ratio_))
    else:
        X_hat = vectors
    current_best_kl = np.inf
    current_best_model = None
    for lr in learning_rates:
        print ('lr: %1.2f' % lr)
        model = TSNE(n_components=2,learning_rate=lr, verbose=1,metric=metric)
        print('Fitting TSNEE...')
        model.fit(X_hat)
        if model.kl_divergence_ < current_best_kl:
            current_best_model = model
            current_best_kl = model.kl_divergence_

    # model = TSNE(n_components=2,learning_rate=6000, verbose=1)

    Y = current_best_model.embedding_
    print('Done fitting TSNEE')
    return Y, current_best_kl, current_best_model

def draw_tsnee_plot(vectors, categories, labels_in_plot=None, dot_size=5, output_file=None, show_plot=True,
                    show_dot_annotations=True, show_legend=True, axis=None):
    if not isinstance(vectors, list):
        raise TypeError('Vectors should be a list of numpy arrays')
    Y,_,_ = fit_tsnee(vectors)

    if labels_in_plot == None:
        labels_in_plot=categories

    num_samples = len(labels_in_plot)

    category2ids = {}
    for i, name in enumerate(categories):
        category2ids.setdefault(name,[])
        category2ids[name].append(i)

    # plt.figure()
    # plt.clf()
    colors = cm.rainbow(np.linspace(0, 1, num_samples))
    if axis is None:
        _, axis = plt.subplots()
    for i, label in enumerate(labels_in_plot):
        x_coord = Y[category2ids[label], 0]
        y_coord = Y[category2ids[label], 1]
        axis.scatter(x_coord,y_coord,
                    label=label + (' (%i)'%len(category2ids[label])),

                    s=[dot_size],
                    # s=[5],
                    color=colors[i])
        if show_dot_annotations:
            for x,y in zip(x_coord.tolist(), y_coord.tolist()):
                axis.annotate(label, (x, y))

    if show_legend:
        plt.legend(loc='upper left', bbox_to_anchor=(0, 0))

    if output_file:
        f = 'tsnee.png'
        plt.savefig(output_file, bbox_inches='tight')
    if show_plot:
        plt.show()
    return axis

if __name__ == '__main__':
    from sklearn import datasets, svm, metrics

    # The digits dataset
    digits = datasets.load_digits()
    rng = np.random.RandomState(0)
    indices = np.arange(len(digits.data))
    rng.shuffle(indices)

    num_points = 10000
    X = digits.data[indices[:num_points]]
    y = digits.target[indices[:num_points]]
    images = digits.images[indices[:num_points]]


    data = X.data.tolist()
    labels = y.flatten().tolist()
    labels = [str(l) for l in labels]

    data = fit_tsnee(data, learning_rates=[1000], metric='cosine')

    draw_tsnee_plot(data,labels, sorted(list(set(labels))))
