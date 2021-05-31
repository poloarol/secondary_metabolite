""" analysis.py -- provides an interface to use several machine learning algorithms """


import os
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass



import pycm
import joblib
import numpy as np
import pandas as pd
import biovec as bv
import tensorflow as tf

from imblearn.over_sampling import SMOTE

from umap import UMAP

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef

from sklearn.impute import KNNImputer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold




@dataclass
class DimensionalReduction():

    """

    Dimensional Reduction provides functions to perform dimensional reduction using umap-learn package.
        https://pair-code.github.io/understanding-umap/
        https://umap-learn.readthedocs.io/en/latest/index.html
    
    """

    seed: int
    data: List

    def __post_init__(self):

        smote = SMOTE(random_state=self.seed, sampling_strategy='not minority') # under sample dataset
        X, y = smote.fit_resample(self.data.iloc[:, 1:], self.data.iloc[:, 0])

        self.train = pd.DataFrame(np.column_stack((y, X)))
    
    def learn(self, in_metric: str = 'euclidean', out_metric: str = 'euclidean', components: int = 3, neighbours: int = 15, distance: float = 0.1) -> UMAP:

        """
        Learns a UMAP dimensional reduction function.

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
                        n_neighbors=neighbours, min_dist=distance)
        
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
    
    def dnn(self) -> tf.keras.models.Sequential:

        # print(self.train.iloc[:, 1:][:5])

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(3000, input_shape=(3, ), activation='relu', name='fc0'))
        model.add(tf.keras.layers.Dense(2000, input_shape=(3, ), activation='relu', name='fc1'))
        model.add(tf.keras.layers.Dense(1500, input_shape=(3, ), activation='relu', name='fc2'))
        model.add(tf.keras.layers.Dense(750, input_shape=(3, ), activation='relu', name='fc3'))
        model.add(tf.keras.layers.Dense(350, input_shape=(3, ), activation='relu', name='fc4'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(5, input_shape=(3, ), activation='softmax', name='fc5'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        y = tf.keras.utils.to_categorical(self.train.iloc[:, 0]) # train
        yy = tf.keras.utils.to_categorical(self.test.iloc[:, 0]) # test

        model.fit(self.train.iloc[:, 1:], y, validation_data=(self.test.iloc[:, 1:], yy), verbose=2, batch_size=10, epochs=200)

        model.evaluate(self.test.iloc[:, 1:], yy)

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
    
    def evaluate_model(self, model: Any, splits: int,  rep: int) -> Tuple[List, List, List, List]:
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
   
    def contigency_table(self, model) -> Tuple[Dict, List]:
        predicted = model.predict(self.test.iloc[:, 1:])

        cm  = pycm.ConfusionMatrix(actual_vector=np.array(self.test.iloc[:, 0]), predict_vector=predicted)

        return cm

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
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (100, 100, 100), (150, 150, 150)],
            'max_iter': [2000, 4000, 6000, 8000]
        }

        cv: int = 10

        # with open(os.path.join(os.getcwd(), 'bgc\\hyperparameters\\hyperparams.txt'), 'a') as f:

        # f.write('=======================================================================')

        estimator = MLPClassifier(random_state=self.seed)

        grid_search = GridSearchCV(estimator=estimator, param_grid=nn_grid, cv=cv, n_jobs=-1)
        grid_search.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])


        print(grid_search.best_params_, grid_search.best_score_)

        print('=======================================================================')

        
        estimator = KNeighborsClassifier()

        grid_search = GridSearchCV(estimator=estimator, param_grid=knn_grid, cv=cv, n_jobs=-1)
        grid_search.fit(self.train.iloc[:, 1:], self.train.iloc[:, 0])


        print(grid_search.best_params_, grid_search.best_score_)

    def evaluate_keras(self, model: Any, splits: int, rep: int) -> Tuple[List, List, List]:
        cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=rep, random_state=self.seed)
        cohen = []
        matt = []
        bal_acc = []

        for test_data, test_id in cv.split(self.test.iloc[:, 1:], self.test.iloc[:, 0]):
            data = self.test.iloc[:, 1:].to_numpy()
            label = self.test.iloc[:, 0].to_numpy()
            y = np.argmax(model.predict(data[test_data]), axis=1)
            class_names = ['PKS', 'NRPS', 'Terpene']
            print(class_names[y])
            # lbl = label[test_data]
            # bal_acc.append(balanced_accuracy_score(lbl, y))
            # cohen.append(cohen_kappa_score(lbl, y))
            # matt.append(matthews_corrcoef(lbl, y))
        
        return bal_acc, cohen, matt


@dataclass
class NLP(object):
    
    filename: str
    model: Any = None

    def __post_init__(self):
        try:
            self.model = bv.models.ProtVec(self.filename, corpus_fname="out_corpus.txt", n=3)
        except FileNotFoundError as e:
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