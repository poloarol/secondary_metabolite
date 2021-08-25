""" analysis.py -- provides an interface to use several machine learning algorithms """


import os
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from scipy.sparse import data
from scipy.sparse.construct import random
from sklearn import neural_network

import pycm
import joblib
import numpy as np
import pandas as pd
import biovec as bv
# import tensorflow as tf

import random

from imblearn.over_sampling import SMOTE

from umap import UMAP
from umap import validation

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import pairwise_distances

from sklearn.impute import KNNImputer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold




@dataclass
class DimensionalReduction(object):

    """

    Dimensional Reduction provides functions to perform dimensional reduction.
        
    
    """

    seed: int
    data: List

    def __post_init__(self):

        smote = SMOTE(random_state=self.seed, sampling_strategy='not minority') # under sample dataset
        X, y = smote.fit_resample(self.data.iloc[:, 1:], self.data.iloc[:, 0])

        self.train = pd.DataFrame(np.column_stack((y, X)))
    
    def umap_learn(self, in_metric: str = 'euclidean', out_metric: str = 'euclidean', components: int = 3, neighbours: int = 15, distance: float = 0.1, weight: float = 0.5) -> UMAP:

        """
        Learns a UMAP dimensional reduction function.
        
        https://pair-code.github.io/understanding-umap/
        https://umap-learn.readthedocs.io/en/latest/index.html

        Parameters
        ----------
            in_metric (str) - Input metric as described by UMAP documentation
            out_meric (str) - Output metric as described by UMAP documentation
            components (int) - Number of components to reduce dataset to
            neighbours (int) - size of local neighbourhood
            distance (float) - effective minimum distance between points

            For more details, visit the UMAP documentation.
                https://umap-learn.readthedocs.io/en/latest/api.html
        
        Returns
        -------
            UMAP model
        """

        reducer = UMAP(random_state=self.seed, init='random', 
                        metric=in_metric, output_metric=out_metric, n_components=components,
                        n_neighbors=neighbours, min_dist=distance, target_weight=weight)
        
        reducer.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return reducer
    
    def combine_union(self, model_1:UMAP, model_2:UMAP) -> UMAP:

        """
        Combine two UMAP models through their union.

        NB: The models should already be trained

        Paramters
        ---------
            model_1 (UMAP) - trained umap model
            model_2 (UMAP) - trained umap model
        
        Return
        ------
            union_model (UMAP) - Union model of both model_1 and model_2
        """

        try:
            
            union_model = model_1 + model_2
            return union_model
        
        except Exception as e:
            raise e('One model is not trained')
    
    def combine_intersection(self, model_1:UMAP, model_2:UMAP) -> UMAP:
        
        """
        Combine two UMAP models through their intersection.

        NB: The models should already be trained

        Paramters
        ---------
            model_1 (UMAP) - trained umap model
            model_2 (UMAP) - trained umap model
        
        Return
        ------
            intersection_model (UMAP) - Intersection model of both model_1 and model_2
        """

        try:
            
            intersection_model = model_1 * model_2
            return intersection_model
        
        except Exception as e:
            raise e('One model is not trained')
    
    def lda_learn(self, solver: str = 'eigen', shrinkage: str = None, components: int = 3) -> LinearDiscriminantAnalysis:
        
        """
        
        Perform a LDA to reduce dimensions.
        
        Excerpt of the technique: 
            The Fisher’s propose is basically to maximize the distance between the mean of each class 
            and minimize the spreading within the class itself. Thus, we come up with two measures: 
            the within-class and the between-class. However, this formulation is only possible 
            if we assume that the dataset has a Normal distribution.
        
        For more: https://towardsdatascience.com/is-lda-a-dimensionality-reduction-technique-or-a-classifier-algorithm-eeed4de9953a
            
        """
        
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, n_components=components)
        lda.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])
        
        return lda

    def nca_learn(self, components: int = 3, init:str = 'auto') -> NeighborhoodComponentsAnalysis:
        
        """
        
        Perform a NCA to reduce dimesions
        
        """
        
        nca = NeighborhoodComponentsAnalysis(n_components=components, init=init, random_state=self.seed)
        nca.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])
        
        return nca
    
    def trustworthiness(self, dataset: pd.DataFrame, embedded: pd.DataFrame, n_neighbors: int, metric: str = 'euclidean'):


        """Expresses to what extent the local structure is retained.
        The trustworthiness is within [0, 1]. It is defined as
        .. math::
            T(k) = 1 - \\frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
                \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))
        where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
        neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
        nearest neighbor in the input space. In other words, any unexpected nearest
        neighbors in the output space are penalised in proportion to their rank in
        the input space.
        * "Neighborhood Preservation in Nonlinear Projection Methods: An
        Experimental Study"
        J. Venna, S. Kaski
        * "Learning a Parametric Embedding by Preserving Local Structure"
        L.J.P. van der Maaten
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        X_embedded : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        n_neighbors : int, optional (default: 5)
            Number of neighbors k that will be considered.
        metric : string, or callable, optional, default 'euclidean'
            Which metric to use for computing pairwise distances between samples
            from the original input space. If metric is 'precomputed', X must be a
            matrix of pairwise distances or squared distances. Otherwise, see the
            documentation of argument metric in sklearn.pairwise.pairwise_distances
            for a list of available metrics.
        Returns
        -------
        trustworthiness : float
            Trustworthiness of the low-dimensional embedding.
        """
        dist_X = pairwise_distances(dataset, metric=metric)
        if metric == 'precomputed':
            dist_X = dist_X.copy()
        # we set the diagonal to np.inf to exclude the points themselves from
        # their own neighborhood
        np.fill_diagonal(dist_X, np.inf)
        ind_X = np.argsort(dist_X, axis=1)
        # `ind_X[i]` is the index of sorted distances between i and other samples
        ind_X_embedded = NearestNeighbors(n_neighbors).fit(embedded).kneighbors(
            return_distance=False)

        # We build an inverted index of neighbors in the input space: For sample i,
        # we define `inverted_index[i]` as the inverted index of sorted distances:
        # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
        n_samples = dataset.shape[0]
        inverted_index = np.zeros((n_samples, n_samples), dtype=int)
        ordered_indices = np.arange(n_samples + 1)
        inverted_index[ordered_indices[:-1, np.newaxis],
                    ind_X] = ordered_indices[1:]
        ranks = inverted_index[ordered_indices[:-1, np.newaxis],
                            ind_X_embedded] - n_neighbors
        t = np.sum(ranks[ranks > 0])
        t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                            (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
        return t


@dataclass
class Clustering(object):
    """
    Clustering class

    data: pd.DataFrame
    """

    data: pd.DataFrame
    

    def hierachical_clustering(self, linkage: str = 'ward'):
        """
        """
        
        random.seed(1024)

        model = AgglomerativeClustering(linkage=linkage, distance_threshold=0.0001, n_clusters=None)
        model.fit(self.data)

        return model


@dataclass
class Supervised(object):
    """
        Provides methods for Supervised Learning

        Parameters
        ----------

        seed: int
        train: pd.DataFrame
        test: pd.DataFrame

    """

    seed: int
    train: Any
    test: Any

    def __post_init__(self):
        
        inputer = KNNImputer()
        train = pd.DataFrame(self.train.iloc[:, 1:])
        test = pd.DataFrame(self.test.iloc[:, 1:])

        if train.isnull().values.any():
            train = inputer.fit_transform(train)
            self.train = pd.DataFrame(np.column_stack((self.train.iloc[:, 0], train)))
        
        if test.isnull().values.any():
            test = inputer.fit_transform(test)
            self.test = pd.DataFrame(np.column_stack((self.test.iloc[:, 0], test)))
        
        smote = SMOTE(random_state=self.seed, sampling_strategy='not majority')
        X, y = smote.fit_resample(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        self.train = pd.DataFrame(np.column_stack((y, X)))

    def rforest(self) -> RandomForestClassifier:
        
        rand_forest = RandomForestClassifier(random_state=self.seed, bootstrap=True, oob_score=True)
        rand_forest.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return rand_forest
 
    def adaRforest(self) -> AdaBoostClassifier:
        ada = AdaBoostClassifier(RandomForestClassifier(random_state=self.seed), algorithm='SAMME.R')
        ada.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return ada
      
    def neural_network(self, hls=(100, 100, 100), activation: str = 'relu', solver: str = 'adam', learning_rate: str = 'constant', max_iter=2000) -> MLPClassifier:
        
        nn = MLPClassifier(random_state=self.seed, max_iter=max_iter, hidden_layer_sizes=hls, activation=activation, solver=solver, learning_rate=learning_rate)
        nn.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return nn
      
    def knn(self, neighbours: int = 2, weights: str = 'uniform', metric: str = 'euclidean', algo: str = 'auto') -> KNeighborsClassifier:

        k_nn = KNeighborsClassifier(n_neighbors=neighbours, weights=weights, metric=metric, algorithm=algo)
        k_nn.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])
        return k_nn
    
    def xgboost(self) -> GradientBoostingClassifier:
        xg = GradientBoostingClassifier()
        xg.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return xg
    

    def save(self, model, filename: str = None) -> None:
        """
        Save estimator to file

        Parameters
        ----------
        model: Supervised Learning Estimator
        filename (str)

        """

        path =  os.path.join(os.getcwd(), 'bgc\\models\\supervised\\{}_.model'.format(filename))
        joblib.dump(model, path)

    
    def evaluate_model(self, model: Any, splits: int,  rep: int) -> Tuple[List, List, List, List]:

        """
            Calculate statistics necessary for multi-classification model evaluation.

            Statistics:
            ----------
            Bal. Accuracy, Accuracy, Cohen's Kappa and Matthews' Correlation Coefficient.

            Parameters:
            ----------
            model: sklearn model to evaluate
            splits: How to split the dataset
            rep: Number of repetition neccessary for model evalution

            Return
            ------

            Tuple(List, List, List, List) -> Tuple(accuarcy, bal. accuracy, cohen kappa, matt. corr. coef)

        """

        cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=rep, random_state=self.seed)
        cohen = []
        matt = []
        acc = cross_val_score(model, self.test.iloc[:, 1:], self.test.iloc[:, 0], scoring='accuracy', cv=cv, n_jobs=-1)
        bal_acc = cross_val_score(model, self.test.iloc[:, 1:], self.test.iloc[:, 0], scoring='balanced_accuracy', cv=cv, n_jobs=-1)

        for test_data, test_id in cv.split(self.test.iloc[:, 1:], self.test.iloc[:, 0]):
            data = self.test.iloc[:, 1:].to_numpy()
            label = self.test.iloc[:, 0].to_numpy()
            y = model.predict(data[test_data])
            lbl = label[test_data]
            cohen.append(cohen_kappa_score(lbl, y))
            matt.append(matthews_corrcoef(lbl, y))

        return acc, bal_acc, cohen, matt
   
    def confusion_matrix(self, model) -> pycm.ConfusionMatrix:

        """
        Generates a confusion matrix, showing how well model fairs for each class.

        Parameters
        ----------
        model: sklearn model to evaluate

        Return:
        ------
        Confusion Matrix produced by pycm
        
        """

        predicted = model.predict(self.test.iloc[:, 1:])

        cm  = pycm.ConfusionMatrix(actual_vector=np.array(self.test.iloc[:, 0]), predict_vector=predicted)

        return cm

    def save(self, name: str, classifier) -> None:

        """

        Save Classifier

        Parameters
        ----------

        name (str): Name of classifier
        classifier: sklearn model to save


        """

        path =  os.path.join(os.getcwd(), 'bgc\\models\\supervised\\{}.model'.format(name))
        joblib.dump(classifier, path)

    def hyperparameter_tuning(self) -> None:

        """

        Tune Hyperparameters to find the best values for model performance

        """

        rf_grid = {
            'bootstrap': [True, False],
            'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            'max_features': ['log2', 'auto', 'sqrt', None],
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        knn_grid = {
            'n_neighbors': [2, 4, 5, 6, 8, 10],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
            'algorithm': ['auto', 'kd_tree', 'ball_tree', 'brute']
        }

        nn_grid = {
            'activation': ['relu', 'tanh', 'logistic', 'identity'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (100, 100, 100), (150, 150, 150)],
            'max_iter': [2000, 4000, 6000, 8000]
        }
        
        xg_grid = {
            'max_depth': range(2, 10, 1),
            'n_estimators': range(60, 220, 40),
            'learning_rate': [0.1, 0.05, 0.01]
        }

        ada_grid = {
            'base_estimator__bootstrap': [True, False],
            'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            'base_estimator__max_features': ['log2', 'auto', 'sqrt', None],
            'base_estimator__criterion': ['gini', 'entropy'],
            'base_estimator__class_weight': [None, 'balanced', 'balanced_subsample'],
            'base_estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'base_estimator__min_samples_split': [2, 5, 10],
            'base_estimator__min_samples_leaf': [1, 2, 4],
        }

        grids = {'rf': [self.rforest(), rf_grid], 'ada': [self.adaRforest(), ada_grid]: 'nn': [self.neural_network(), nn_grid], 'xg': [self.xgboost(), xg_grid], 'knn': [self.knn(), knn_grid]}

        for key, value in grids:
            gridcv = GridSearchCV(value[0], param_grid=value[1])
            gridcv.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

            print(key, gridcv.best_params_, gridcv.best_score_)

@dataclass
class NLP(object):

    """
    Assist in training biovec model to perform NLP.

    Parameters
    ----------

    filename: DB to create embedding from i.e. learn vectorization
    model: None


    """
    
    filename: str
    model: Any = None

    def __post_init__(self):
        try:
            self.model = bv.models.ProtVec(self.filename, corpus_fname="out_corpus.txt", n=3)
        except FileNotFoundError as e:
            print(e.__repr__())
        
        np.random.seed(1042)

    def get(self):

        """ Provide trained biovec model """
        return self.model
    
    def save(self, filename: str):
        """ Save biovec model to file """
        path: str = os.path.join(os.getcwd(), 'bgc\\models\\biovec\\{0}.model'.format(filename))
        self.model.save(path)
    
    def load(self, filename: str):
        """ Load biovec model into workspace """
        return bv.models.load_protvec(os.path.join(os.getcwd(), 'bgc\\models\\biovec\\{0}.model'.format(filename)))




"""

    @article{asgari2015continuous,
    title={Continuous Distributed Representation of Biological Sequences for Deep Proteomics and Genomics},
    author={Asgari, Ehsaneddin and Mofrad, Mohammad RK},
    journal={PloS one},
    volume={10},
    number={11},
    pages={e0141287},
    year={2015},
    publisher={Public Library of Science}
    }

    @article{2018arXivUMAP,
        author = {{McInnes}, L. and {Healy}, J. and {Melville}, J.},
        title = "{UMAP: Uniform Manifold Approximation
        and Projection for Dimension Reduction}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1802.03426},
        primaryClass = "stat.ML",
        keywords = {Statistics - Machine Learning,
                    Computer Science - Computational Geometry,
                    Computer Science - Learning},
        year = 2018,
        month = feb,
    }
"""