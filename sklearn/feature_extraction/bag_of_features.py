""" Bag-of-Features (BOF) histogram representation

 Module to extract vocabularies of "visual words" from local features
 and quantize them to create a BOF (histogram of occurences of visual words)

"""

# Authors: Adrien Gaidon <adnothing@gmail.com>


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
                whether the BOf models are outputed as sparse CSR vectors

    Methods
    -------

    fit(X):
        compute the vocabulary using MiniBatchKMeans

    transform(X):
        compute the BOF histograms

    predict(X):
        get the closest visual word for each feature

    Attributes
    ----------

    voc_model_: MiniBatchKMeans instance,
                the model obtained when learning the vocabulary

    visual_words_: array, [n_clusters, n_features]
        Coordinates of visual words (cluster centers)
    """

    def __init__(self, voc_size=1000, normalize=True, sparse_out=True):
        self.voc_size = voc_size
        self.normalize = normalize
        self.sparse_out = sparse_out

    def fit(self, X, y=None):
        """ Compute the visual vocabulary using MiniBatchKMeans

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_points, d]
           A set of n_points local d-dimensional features

        Returns
        -------
        self
        """
        self.voc_model_ = MiniBatchKMeans(k=self.voc_size, init='k-means++',
                                          max_iter=100, chunk_size=1000)
        self.voc_model_.fit(X)
        self.visual_words_ = self.voc_model_.cluster_centers_
        return self

    def predict(self, X, y=None):
        """Predict the closest visual word each sample in X corresponds to.

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
            raise AttributeError("Model has not been trained yet. ")
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
           BOF histogram counting the occurences of visual words
           it is L1-normalized if (self.normalize == True)
        """
        # assign each feature to its closest visual word
        labels = self.predict(X)
        # build the histogram
        b = np.bincount(labels, minlength=self.voc_size).squeeze().astype(np.float)
        # normalize it if requested
        if self.normalize:
            b /= b.sum()
        # transform it to a sparse vector if requested
        if self.sparse_out:
            b = csr_matrix(b)
        return b
