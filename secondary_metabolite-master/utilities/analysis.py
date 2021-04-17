""" analysis.py -- provides an interface to use several machine learning algorithms """

import os
import random
import re
from statistics import mode

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import biovec as bv

from imblearn.over_sampling import SMOTE

from umap import UMAP


from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score


from sklearn.impute import KNNImputer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

import tensorflow as tf
# from tensorflow.keras.layers import Embedding, Dense, LSTM
# from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam



@dataclass
class DimensionalReduction(object):
    """
        Provides methods to run Dimensional Reduction
    """

    seed: int
    data: List

    def __post_init__(self):

        smote = SMOTE(random_state=self.seed, sampling_strategy='not minority') # under sample dataset
        X, y = smote.fit_resample(self.data.iloc[:, 1:], self.data.iloc[:, 0])

        self.train = pd.DataFrame(np.column_stack((y, X)))


    def pca(self, n_components: int=50) -> pd.DataFrame:
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

        Pandas DataFrame
        """

        pca = PCA(n_components=n_components, random_state=self.seed, whiten=True)
        embedding = pca.fit_transform(self.train.iloc[:, 1:])

        if n_components == 2:
            data = np.column_stack([np.round(self.train.iloc[:, 0], decimals=0), np.round(embedding, decimals=5)])
        data = pd.DataFrame(np.column_stack([self.train.iloc[:, 0], embedding]))
        self.save(pca, 'pca_{}'.format(n_components))

        return data
    
    def scree_plot(self) -> np.array:

        """
            Generates np array of Principal Components and their relative importance.
        """

        A = np.asmatrix(self.train.iloc[:, 1:].T) * np.asmatrix(self.train.iloc[:, 1:])
        U, S, V = np.linalg.svd(A)

        pcs = S**2/np.sum(S**2)

        return pcs

    def umap(self, input_metric: str='euclidean', neighbours: int = 15, min_dist: float = 0.1, components: int = 3, output_metric: str='euclidean', weight:float = 0.5) -> Tuple[UMAP, pd.DataFrame]:
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


        reducer = UMAP(metric=input_metric, random_state=self.seed, n_neighbors=neighbours, min_dist=min_dist,
                            n_components=components, output_metric=output_metric, target_weight=weight, force_approximation_algorithm=True,
                            transform_seed=self.seed, init='random')
        embedding = reducer.fit_transform(self.train.iloc[:, 1:], self.train.iloc[:, 0])
        data = pd.DataFrame(np.column_stack((self.train.iloc[:, 0], embedding)), index=None)

        return reducer, data

    def export_results(self, embedding, filename: str) -> None:
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

    def save(self, reducer, filename: str = None) -> None:
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
    
    def transform(self, model: Any, data: pd.DataFrame) -> pd.DataFrame:

        """
            Using a trained model, transform data into the desired properties

            Parameters
            ----------

            model: Trained Dimensional Reduction algorithm
            data (pd.DataFrame): Data to be transformed

            Returns:
            -------

            Transformed pd.DataFrame of data

            Raises (Exception)
            ------------------

            If data is empty
        """
        
        try:
            embedding = model.transform(data)
            embedding = pd.DataFrame(embedding, index=None)

            return embedding
        except Exception as e:
            raise e


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
        
        # smote = SMOTE(random_state=self.seed, sampling_strategy='not majority')
        # X, y = smote.fit_resample(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        # self.train = pd.DataFrame(np.column_stack((y, X)))

    def rforest(self) -> RandomForestClassifier:
        
        rand_forest = RandomForestClassifier(random_state=self.seed, bootstrap=True, oob_score=True)
        rand_forest.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return rand_forest
 
    def adaRforest(self) -> AdaBoostClassifier:
        ada = AdaBoostClassifier(RandomForestClassifier(random_state=self.seed), algorithm='SAMME.R')
        ada.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return ada
      
    def neural_network(self, hls=(100, 100, 100)) -> MLPClassifier:
        nn = MLPClassifier(random_state=self.seed, max_iter=4000, hidden_layer_sizes=hls)
        nn.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return nn
      
    def knn(self) -> KNeighborsClassifier:

        k_nn = KNeighborsClassifier()
        k_nn.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])
        return k_nn

    
    def xgboost(self) -> GradientBoostingClassifier:
        xg = GradientBoostingClassifier()
        xg.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])

        return xg
    
    def dnn(self):

        # print(self.train.iloc[:, 1:][:5])

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1500, input_shape=(3, ), activation='relu', name='fc0'))
        model.add(tf.keras.layers.Dense(750, input_shape=(3, ), activation='relu', name='fc1'))
        model.add(tf.keras.layers.Dense(300, input_shape=(3, ), activation='relu', name='fc2'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(3, input_shape=(3, ), activation='softmax', name='fc3'))

        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # y = tf.keras.utils.to_categorical(self.train.iloc[:, 0], num_classes=3)

        model.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0], validation_data=(self.test.iloc[:, 1:], self.test.iloc[:, 0]), verbose=2, batch_size=5, epochs=200)

        model.evaluate(self.test.iloc[:, 1:], self.test.iloc[:, 0])

        return model

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
    
    def evaluate_model(self, model, splits,  rep) -> Tuple:
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
   
    def contigency_table(self, model) -> Tuple:
        predicted = model.predict(self.test.iloc[:, 1:])

        # tmp = model.predict_proba(self.test.iloc[:, 1:])

        con_matrix = confusion_matrix(self.test.iloc[:, 0], predicted, labels=[0, 1, 4])

        FP = con_matrix.sum(axis=0) - np.diag(con_matrix)
        FN = con_matrix.sum(axis=1) - np.diag(con_matrix)
        TP = np.diag(con_matrix)
        TN = con_matrix.sum() - (FP + FN + TP)

        con_table = {'FP': FP.sum(), 'FN': FN.sum(), 'TP': TP.sum(), 'TN': TN.sum()}

        return con_table, con_matrix

    def save(self, name: str, classifier) -> None:
        path =  os.path.join(os.getcwd(), 'bgc\\models\\supervised\\{}.model'.format(name))
        joblib.dump(classifier, path)

    def hyperparameter_tuning(self) -> None:

        # scaler = StandardScaler()
        # self.train.iloc[:, 1:] = scaler.fit_transform(self.train.iloc[:, 1:])

        rf_grid = {
            'bootstrap': [True, False],
            'n_estimators': [2, 10, 100, 1000],
            'max_features': ['log2', 'auto', 'sqrt', None],
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced', 'balanced_subsample']
            # 'max_depth': np.linspace(1, 1024, 128, endpoint=True),
            # 'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
            # 'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True)
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
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (100, 100, 100), (150, 150, 150)]
        }

        cv: int = 10

        # with open(os.path.join(os.getcwd(), 'bgc\\hyperparameters\\hyperparams.txt'), 'a') as f:

        # f.write('=======================================================================')

        estimator = MLPClassifier(random_state=self.seed, max_iter=4000)

        grid_search = GridSearchCV(estimator=estimator, param_grid=nn_grid, cv=cv, n_jobs=-1)
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
"""