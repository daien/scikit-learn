""" Bag-of-Features (BOF) histogram representation

Module to extract a "vocabulary" (also called "code book") of local features
and quantize sets of such local features to create a BOF model
(histogram of occurences of words from the learned vocabulary).

"""

# Authors: Adrien Gaidon <adnothing@gmail.com>
# License: BSD


import numpy as np
from scipy.sparse import csr_matrix


from ..base import BaseEstimator
from ..cluster import MiniBatchKMeans


class BagOfFeatures(BaseEstimator):
    """ Bag-of-Features (BOF) extractor

    Parameters
    ----------
    voc_size: int, optional, default: 1000,
              vocabulary size (number of visaul words)

    normalize: boolean, optional, default: True,
               whether to L1-normalize the BOFs

    sparse_out: boolean, optional, default: True,
                whether the BOF models are outputed as sparse CSR vectors

    Methods
    -------
    fit(X):
        compute the vocabulary using MiniBatchKMeans

    transform(X):
        compute the BOF histograms

    predict(X):
        get the closest word for each feature

    Attributes
    ----------
    voc_model_: MiniBatchKMeans instance,
                the model obtained when learning the vocabulary

    vocabulary_: array, [n_clusters, n_features]
                 coordinates of the cluster centers

    """

    def __init__(self, voc_size=1000, normalize=True, sparse_out=True,
                 init='k-means++', max_iter=100, chunk_size=1000):
        self.voc_size = voc_size
        self.normalize = normalize
        self.sparse_out = sparse_out
        self.init = init
        self.max_iter = max_iter
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        """ Compute the vocabulary using MiniBatchKMeans

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_points, d]
           A set of n_points local d-dimensional features

        Returns
        -------
        self

        """
        self.voc_model_ = MiniBatchKMeans(k=self.voc_size,
                                          init=self.init,
                                          max_iter=self.max_iter,
                                          chunk_size=self.chunk_size,
                                          compute_labels=False).fit(X)
        self.vocabulary_ = self.voc_model_.cluster_centers_
        return self

    def predict(self, X, y=None):
        """Predict the closest word each sample in X corresponds to.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_points, d]
           A set of n_points local d-dimensional features

        Returns
        -------
        Y : array, shape [n_samples,]
            Index of the closest center each sample belongs to.

        """
        if not hasattr(self, "voc_model_"):
            raise AttributeError("Model has not been trained yet.")
        return self.voc_model_.predict(X)

    def transform(self, X, y=None):
        """Transform a set of local features to a BOF histogram

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_points, d]
           A set of n_points local d-dimensional features

        Returns
        -------
        b: {array-like, sparse matrix}, shape [voc_size, ]
           BOF histogram counting the occurences of words
           it is L1-normalized if (self.normalize == True)

        """
        # assign each feature to its closest word
        labels = self.predict(X)
        # build the histogram
        b = np.bincount(labels, minlength=self.voc_size).squeeze().astype(np.float)
        # normalize it if requested
        if self.normalize:
            bs = b.sum()
            if bs > 0:
                b /= bs
        # transform it to a sparse vector if requested
        if self.sparse_out:
            b = csr_matrix(b)
        return b
