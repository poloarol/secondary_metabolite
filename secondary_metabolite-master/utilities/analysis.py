""" analysis.py -- provides an interface to use several machine learning algorithms """

import os
import random
import re

from typing import Any, List
from dataclasses import dataclass
from joblib import parallel
from numpy.lib import column_stack
from pyfasta.fasta import FastaNotFound
from scipy import stats
from scipy.optimize.zeros import results_c
from scipy.sparse.construct import random
from sklearn.utils import shuffle

import joblib
import numpy as np
import pandas as pd
import biovec as bv

from scipy.stats import chi2
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

from imblearn.over_sampling import SMOTE

from umap import UMAP

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.impute import KNNImputer

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.cluster import KMeans

@dataclass
class DimensionalReduction(object):
    """
        Provides methods to run Dimensional Reduction
    """

    seed: int
    data: List

    def __post_init__(self):

        smote = SMOTE(random_state=self.seed, sampling_strategy='not majority')
        X, y = smote.fit_resample(self.data.iloc[:, 1:], self.data.iloc[:, 0])

        self.train = pd.DataFrame(np.column_stack((y, X)))


    def pca(self, n_components: int=50):
        """
        PCA: Principal Component Analysis.

        Allows to run PCA - deterministic dimensional reduction algorithm, to convert high dimensional
        dataset to a lower dimensional dataset

        The resulting data points, maximize variance within the dataset

        Parameters
        ----------

        n_components: int = 50 (default)

        Returns
        -------

        Pandas Array of data points
        """

        pca = PCA(n_components=n_components, random_state=self.seed, whiten=True)
        embedding = pca.fit_transform(self.train.iloc[:, 1:])

        if n_components == 2:
            data = np.column_stack([np.round(self.train.iloc[:, 0], decimals=0), np.round(embedding, decimals=5)])
        data = np.column_stack([self.train.iloc[:, 0], embedding])
        self.save(pca, 'pca_{}'.format(n_components))
        return data
    
    def scree_plot(self):

        A = np.asmatrix(self.train.iloc[:, 1:].T) * np.asmatrix(self.train.iloc[:, 1:])
        U, S, V = np.linalg.svd(A)

        pcs = S**2/np.sum(S**2)

        return pcs

    def umap(self, input_metric: str='euclidean', neighbours: int = 15, min_dist: float = 0.1, components: int = 3, output_metric: str='euclidean', name:str = '', weight:float = 0.5, spread:float = 1):
        """
        Runs UMAP from the umap-learn package McInness - 2018 v0.4

        Parameters 
        ----------
        input_metric: str = 'euclidean
        neighbours: int = 15
        min_dist: float = 0.1
        components: int = 2
        output_metric: str = 'euclidean'

        For description of package parameters & understanding of UMAP visit: 
        1. https://github.com/lmcinnes/umap/tree/0.5dev
        2. https://pair-code.github.io/understanding-umap/

        Return
        -------
        reducer (estimator)

        """


        reducer = UMAP(metric=input_metric, random_state=self.seed, n_neighbors=neighbours, min_dist=min_dist, spread=spread,
                            n_components=components, output_metric=output_metric, target_weight=weight, force_approximation_algorithm=True,
                            transform_seed=self.seed, n_jobs=1)
        results = reducer.fit_transform(self.train.iloc[:, 1:], self.train.iloc[:, 0])
        data = pd.DataFrame(np.column_stack((self.train.iloc[:, 0], results)), index=None)

        # self.export_results(data, 'umap_train_{}_{}_{}_{}_{}'.format(input_metric, output_metric, min_dist, neighbours, weight))

        return reducer, data

    def export_results(self, embedding, filename: str):
        """
        Write Dimensional Reduction embedding to file i.e. Fitted model

        Parameters
        ----------
        embedding: LDA, NCA and UMAP embedding
        algorithm (str): specification of algorithm
        filename (str): optional filename

        """

        embedding = pd.DataFrame(embedding)
        embedding.to_csv(os.path.join(os.getcwd(), 'bgc\\embedding\\{0}.csv'.format(filename)), sep=',', index=False)

    def save(self, reducer, filename: str = None):
        """
        Save estimator to file

        Parameters
        ----------
        reducer: Dimensional Reduction Estimator
        in_metric (str): Metric used to embbed the vectorial space
        oput_metric (str): Projection metric


        """

        path =  os.path.join(os.getcwd(), 'bgc\\models\\unsupervised\\{}.model'.format(filename))
        joblib.dump(reducer, path)
    
    def transform(self, model, data):
        
        try:
            embedding = model.transform(data.iloc[:, 1:])
            embedding = pd.DataFrame(np.column_stack((data.iloc[:, 0], embedding)), index=None)
            # self.export_results(embedding, '{0}'.format(filename))

            return embedding
        except Exception as e:
            return pd.DataFrame([])

    


@dataclass
class Supervised(object):
    """
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

    def rforest(self):
        
        rand_forest = RandomForestClassifier(random_state=self.seed, bootstrap=True, oob_score=True)
        rand_forest.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return rand_forest
 
    def adaRforest(self, class_weight: str=None, criterion: str='gini', features: str='auto', estimators: int=100):
        ada = AdaBoostClassifier(RandomForestClassifier(random_state=self.seed), algorithm='SAMME.R')
        ada.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return ada
      
    def neural_network(self, lr: str = 'constant', slv: str = 'lbfgs', activation:str = 'relu', hls=(100,)):
        nn = MLPClassifier(random_state=self.seed, max_iter=4000)
        nn.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return nn
      
    def knn(self, algorithm: str='auto', metric: str='minkowski', neighbor: int=5, weight: str='uniform'):

        k_nn = KNeighborsClassifier()
        k_nn.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])
        return k_nn
    
    def xgboost(self):
        xg = GradientBoostingClassifier()
        xg.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return xg

    def save(self, model, filename: str = None):
        """
        Save estimator to file

        Parameters
        ----------
        model: Supervised Learning Estimator
        filename (str)

        """

        path =  os.path.join(os.getcwd(), 'bgc\\models\\supervised\\{}_.model'.format(filename))
        joblib.dump(model, path)
    
    def evaluate_model(self, model, splits, rep):
        cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=rep, random_state=self.seed)
	    # evaluate the model and collect the results
        scores = cross_val_score(model, self.test.iloc[:, 1:], self.test.iloc[:, 0], scoring='accuracy', cv=cv, n_jobs=-1)
        
        return scores
    
    def contigency_table(self, model):
        predicted = model.predict(self.test.iloc[:, 1:])

        # tmp = model.predict_proba(self.test.iloc[:, 1:])
        # print(tmp)
        # a = model.decision_function(self.test.iloc[:, 1:])
        # print(a)

        con_matrix = confusion_matrix(self.test.iloc[:, 0], predicted, labels=[0, 1, 2, 3, 4, 5])

        FP = con_matrix.sum(axis=0) - np.diag(con_matrix)
        FN = con_matrix.sum(axis=1) - np.diag(con_matrix)
        TP = np.diag(con_matrix)
        TN = con_matrix.sum() - (FP + FN + TP)

        con_table = {'FP': FP.sum(), 'FN': FN.sum(), 'TP': TP.sum(), 'TN': TN.sum()}

        return con_table, con_matrix
    
    def roc_curves(self): 
        pass
 
    def mcnemar(self, contigency_table):
        
        stat, p, dof, expected = chi2_contingency(contigency_table)
        prob = 0.95

        critical = chi2.ppf(prob, dof)
        
        if abs(stat) >= critical:
	        print('Dependent (reject H0)')
        else:
	        print('Independent (fail to reject H0)')
        # interpret p-value
        alpha = 1.0 - prob
        print('significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')

    def save(self, name: str, classifier):
        path =  os.path.join(os.getcwd(), 'bgc\\models\\supervised\\{}.model'.format(name))
        joblib.dump(classifier, path)

    def hyperparameter_tuning(self):

        scaler = StandardScaler()
        self.train.iloc[:, 1:] = scaler.fit_transform(self.train.iloc[:, 1:])

        rf_grid = {
            'estimator__bootstrap': [True, False],
            'estimator__n_estimators': [4500, 8500, 10000, 15000, 20000],
            'estimator__max_features': ['log2', 'auto', 'sqrt', None],
            'estimator__criterion': ['gini', 'entropy'],
            'estimator__class_weight': [None, 'balanced', 'balanced_subsample'],
            'estimator__max_depth': np.linspace(1, 1024, 128, endpoint=True),
            'estimator__min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
            'estimator__min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True)
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
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)]
        }

        cv: int = 10

        # with open(os.path.join(os.getcwd(), 'bgc\\hyperparameters\\hyperparams.txt'), 'a') as f:

        # f.write('=======================================================================')

        estimator = OneVsRestClassifier(RandomForestClassifier(random_state=self.seed, oob_score=True), n_jobs=-1)

        grid_search = GridSearchCV(estimator=estimator, param_grid=rf_grid, cv=cv, n_jobs=-1)
        grid_search.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])


        print(grid_search.best_params_, grid_search.best_score_)

        print('=======================================================================')



@dataclass
class NLP(object):
    
    filename: str
    model: Any = None

    def __post_init__(self):
        try:
            self.model = bv.models.ProtVec(self.filename, corpus_fname="out_corpus.txt", n=3)
        except (FileNotFoundError, FastaNotFound) as e:
            print(e.__repr__())
        
        np.random.seed(1042)

    def get(self):
        return self.model
    
    def save(self, filename: str):
        path: str = os.path.join(os.getcwd(), 'bgc\\models\\biovec\\{0}.model'.format(filename))
        self.model.save(path)
    
    def load(self, filename: str):
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

    @article {NBC2020,
        author = {Narayan, Ashwin and Berger, Bonnie and Cho, Hyunghoon},
        title = {Density-Preserving Data Visualization Unveils Dynamic Patterns of Single-Cell Transcriptomic Variability},
        journal = {bioRxiv},
        year = {2020},
        doi = {10.1101/2020.05.12.077776},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2020/05/14/2020.05.12.077776},
        eprint = {https://www.biorxiv.org/content/early/2020/05/14/2020.05.12.077776.full.pdf},
    }

    @article {NBC2020,
        author = {Sainburg, Tim and McInnes, Leland and Gentner, Timothy Q.},
        title = {Parametric UMAP: learning embeddings with deep neural networks for representation and semi-supervised learning},
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {},
        primaryClass = "stat.ML",
        keywords = {Statistics - Machine Learning,
                    Computer Science - Computational Geometry,
                    Computer Science - Learning},
        year = 2020,
    }

    @article {mcinnes2017hdbscan,
        title = {hdbscan: Hierarchical density based clustering},
        author = {McInnes, Leland and Healy, John and Astels, Steve},
        journal = {The Journal of Open Source Software},
        volume = {2},
        number = {11},
        pages = {205},
        year = {2017}
    }

    @misc{riese2019susicode,
    author = {Riese, Felix~M.},
    title = {{SuSi: Supervised Self-Organizing Maps in Python}},
    year = {2019},
    DOI = {10.5281/zenodo.2609130},
    publisher = {Zenodo},
    howpublished = {\href{https://doi.org/10.5281/zenodo.2609130}{doi.org/10.5281/zenodo.2609130}}
}

"""