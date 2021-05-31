""" app.py """

import os
import gzip
import time
import joblib
import shutil
import random
import argparse
import itertools
import statistics

from typing import List
from multiprocessing import Pool

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from utilities.download import Scraper
from utilities.reader import ReadGB
from utilities.analysis import NLP
from utilities.analysis import Supervised
from utilities.analysis import DimensionalReduction


def learn(metric):
    """

        Optimize UMAP by varying the metrics and neighbours. 
        The UMAP optimization is tested using supervised learning.
        The optmization results are written to a file, with each containing
            accuracy, balanced accuracy, Cohen's Kappa and Matt. Corr. Coef

        Parameters
        ----------
        metric (tuple[str, str]) e.g. (euclidean, euclidean)
    """

    seed, train = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\train.csv'))
    seed, test = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\test.csv'))

    scaler = RobustScaler().fit(train.iloc[:, 1:])

    train = np.column_stack((train.iloc[:, 0], scaler.transform(train.iloc[:, 1:])))
    test = np.column_stack((test.iloc[:, 0], scaler.transform(test.iloc[:, 1:])))

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    dim_reduction = DimensionalReduction(seed, train)

    neighbours: List = [5, 15, 30, 45, 60, 75, 90]

    for neighbour in neighbours:

        print('UMAP-learn has begun - num. of neighbours: {}'.format(neighbour))

        umap = dim_reduction.learn(in_metric=metric[0], out_metric=metric[1], neighbours=neighbour)

        transformed_train = np.column_stack((train.iloc[:, 0], umap.transform(train.iloc[:, 1:])))
        transformed_test = np.column_stack((test.iloc[:, 0], umap.transform(test.iloc[:, 1:])))

        transformed_train = pd.DataFrame(transformed_train)
        transformed_test = pd.DataFrame(transformed_test)

        print('Evaluating UMAP model')

        evaluate_umap_model(seed, transformed_train, transformed_test)

        print('UMAP evaluation ended')

        print('UMAP-learn has ended - num. of neighbours: {}'.format(neighbour))

    # evaluate umap models using supervised learning


def evaluate_umap_model(seed: int, train: pd.DataFrame, test: pd.DataFrame) -> None:

    supervised = Supervised(seed, train, test)

    models = {}

    models['rf'] = supervised.rforest()
    models['ada_rf'] = supervised.adaRforest()
    models['nn'] = supervised.neural_network()
    models['knn'] = supervised.knn()
    models['xg'] = supervised.xgboost()

    for key, model in models.items():
        accuracy, bal_accuracy, cohen_kappa, matt_corr_coef = supervised.evaluate_model(model, 3, 5)
        cm = supervised.contigency_table(model)

        print('Classifier: {}'.format(key))
        print('Accuracy: {} +/- {}'.format(statistics.mean(accuracy), statistics.stdev(accuracy)))
        print('Bal. Accuracy: {} +/- {}'.format(statistics.mean(bal_accuracy), statistics.stdev(bal_accuracy)))
        print("Cohen's Kappa: {} +/- {}".format(statistics.mean(cohen_kappa), statistics.stdev(cohen_kappa)))
        print("Matthew's Corr. Coef: {} +/- {}".format(statistics.mean(matt_corr_coef), statistics.stdev(matt_corr_coef)))

        cm.relabel(mapping={0: 'PKS', 1: 'NRPS', 3: 'Terpene', 4: 'Bacteriocin'})

        cm.print_matrix()

def split(filename: str):

    """Split dataset into training and testing set """
    seed, data = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\{0}'.format(filename)))
    tmp = data.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(tmp, test_size=0.4, random_state=seed)

    np.savetxt(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\train.csv'), train, comments='', delimiter=',')
    np.savetxt(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\test.csv'), test, comments='', delimiter=',')

def classify(dim: int = 3):

    seed, nan = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\train.csv'))
    seed, train = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\validation.csv'))
    seed, test = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\mibig\\4_class_mb.csv'))

    # scaler = joblib.load(os.path.join(os.getcwd(), 'bgc\\models\\scaler\\RobustScaler.model'))
    scaler = RobustScaler()
    scaler.fit(nan.iloc[:, 1:])

    scaled_train = np.column_stack((train.iloc[:, 0], scaler.transform(train.iloc[:, 1:])))
    scaled_train = pd.DataFrame(scaled_train)
    scaled_test = np.column_stack((test.iloc[:, 0], scaler.transform(test.iloc[:, 1:])))
    scaled_test = pd.DataFrame(scaled_test)

    dim_reduction = DimensionalReduction(seed, scaled_train)

    umap, train = dim_reduction.umap(neighbours=75, input_metric='chebyshev', output_metric='chebyshev')
    test_transformed = dim_reduction.transform(umap, scaled_test.iloc[:, 1:])

    test = pd.DataFrame(np.column_stack((test.iloc[:, 0], test_transformed)))

    models = {}

    supervised = Supervised(seed, train, test)

    models['rf'] = supervised.rforest()
    models['ada_rf'] = supervised.adaRforest()
    models['knn'] = supervised.knn()
    models['nn'] = supervised.neural_network()

    for key, model in models.items():
        if key != 'dnn':
            acc, bal_acc, cohen, matt = supervised.evaluate_model(model, 3, 5)
            print('{}'.format(key))
            print('Accuracy: {} +/- {}'.format(round(statistics.mean(acc), 3), round(statistics.stdev(acc), 3)))
            print('Bal. Accuracy: {} +/- {}'.format(round(statistics.mean(bal_acc), 3) , round(statistics.stdev(bal_acc), 3)))
            print("Cohen's Kappa: {} +/- {}".format(round(statistics.mean(cohen), 3) , round(statistics.stdev(cohen), 3)))
            print('Matt. Corr. Coef: {} +/- {}'.format(round(statistics.mean(matt), 3) ,round(statistics.stdev(matt), 3)))

            a, b = supervised.contigency_table(model)
            print(b)

def tuning(neighbour: int = 90, input_metric: str = 'chebyshev', output_metric: str = 'chebyshev'):  

    seed, nan = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\train.csv'))
    seed, train = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\validation.csv'))
    seed, test = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\mibig\\4_class_mb.csv'))

    scaler = RobustScaler()
    scaler.fit(nan.iloc[:, 1:])

    scaled_train = np.column_stack((train.iloc[:, 0], scaler.transform(train.iloc[:, 1:])))
    scaled_train = pd.DataFrame(scaled_train)
    scaled_test = np.column_stack((test.iloc[:, 0], scaler.transform(test.iloc[:, 1:])))
    scaled_test = pd.DataFrame(scaled_test)

    dim_reduction = DimensionalReduction(seed, scaled_test)

    umap, train = dim_reduction.umap(neighbours=neighbour, input_metric=input_metric, output_metric=output_metric)
    test_transformed = dim_reduction.transform(umap, scaled_test.iloc[:, 1:])

    test = pd.DataFrame(np.column_stack((test.iloc[:, 0], test_transformed)))

    supervised = Supervised(seed, train, test)
    supervised.hyperparameter_tuning()

def shuffle(filename: str):
    """ shuffle the pandas dataset and divide it into a 4 (train) : 1 (test) ratio. """

    seed: int = 42
    data = pd.read_csv(filename, low_memory=True, header=None)
    data = clean_dataset(data)
    data = data.sample(frac=1).reset_index(drop=True)
    return seed, data

def clean_dataset(df):
    print('start - cleaning dataset')
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    print('done - cleaning dataset')
    return df[indices_to_keep].astype(np.float64)

def mibig(file):
    """ access the mibig database, download genbank files and convert them to vectors"""
    
    data = np.array([])

    if os.path.isfile(os.path.join(os.getcwd(), 'bgc\\mibig\\nps\\{0}'.format(file))):
        with open(os.path.join(os.getcwd(), 'bgc\\mibig\\nps\\{0}'.format(file))) as lines:
            for line in lines:
                scraper = Scraper(line.strip())
                scraper.mibig_download()
                path = os.path.join(os.getcwd(), 'tmp\\gbk\\{}.gbk'.format(line.strip()))
                rb = ReadGB(path)
                # rb.to_fasta(os.path.join(os.getcwd(), 'bgc\\mibig\\bgcs\\{}.fasta'.format(line.strip())), line.strip())
                vector = rb.get_vector()
                vector = vector.flatten()

                if vector.shape[0] == 300:
                    if data.size == 0:
                        data = np.array([vector])
                    elif vector.size != 0:
                        data = np.concatenate((data, np.array([vector])))
                    
        np.savetxt('bgc\\mibig\\np_vecs\\{}_uniprot2vec.csv'.format(file.split('.')[0]), data, delimiter=',')

def antismash(file):
    """ Access AntiSmash DB, download datasets and write them as both genbank and fasta. """
    
    data = np.array([])

    with open(os.path.join(os.getcwd(), 'bgc\\antismash\\antismash\\{}'.format(file))) as lines:
        for line in lines:
            scaper = Scraper()
            time.sleep(3)
            success = scaper.antismash_download(line.strip())
            if success:
                p = line.strip().split('/')[7] + '_' + success
                path = os.path.join(os.getcwd(), 'tmp/gbk/{}.gbk'.format(p))
                rb = ReadGB(path)
                # rb.to_fasta(os.path.join(os.getcwd(), 'bgc\\antismash\\bgcs\\{0}'.format(p)))
                vector = rb.get_vector()
                vector = vector.flatten()

                if vector.shape[0] == 300:
                    if data.size == 0:
                        data = np.array([vector])
                    elif vector.size != 0:
                        data = np.concatenate((data, np.array([vector])))
    np.savetxt('bgc\\antismash\\vectors\\{}_uniprot2vec.csv'.format(file.split('.')[0]), data, delimiter=',')

def generate_models():

    """
        Generate final models and save them.
    """

    seed, train_antismash = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\train.csv'))
    seed, test_antismash = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\validation.csv'))
    seed, mibig = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\4_class_mb.csv'))

    scaler = RobustScaler()
    scaler.fit(train_antismash.iloc[:, 1:])

    joblib.dump(scaler, os.path.join(os.getcwd(), 'bgc\\models\\RobustScaler.model'))

    train_antismash = np.column_stack((train_antismash.iloc[:, 0].astype(int), scaler.transform(train_antismash.iloc[:, 1:])))
    test_antismash = np.column_stack((test_antismash.iloc[:, 0].astype(int), scaler.transform(test_antismash.iloc[:, 1:])))

    dimreduction = DimensionalReduction(seed, train_antismash)
    umap_1, data_1 = dimreduction.umap(input_metric='chebyshev', neighbours=90)
    dimreduction.export_results(data_1, 'embedding_chebyshev_euclidean_90.csv')
    umap_2, data_2 = dimreduction.umap(input_metric='chebyshev', output_metric='manhattan', neighbours=75)
    dimreduction.export_results(data_2, 'embedding_chebyshev_manhattan_75.csv')
    umap_3, data_3 = dimreduction.umap(input_metric='chebyshev', output_metric='chebyshev', neighbours=75)
    dimreduction.export_results(data_3, 'embedding_chebyshev_chebyshev-75.csv')

    joblib.dump(umap_1, os.path.join(os.getcwd(), 'bgc\\models\\UMAP_Chebyshev_Euclidean.model'))
    joblib.dump(umap_2, os.path.join(os.getcwd(), 'bgc\\models\\UMAP_Chebyshev_Manhattan.model'))
    joblib.dump(umap_3, os.path.join(os.getcwd(), 'bgc\\models\\UMAP_Chebyshev_Chebyshev.model'))

    supervised = Supervised(seed, train_antismash, test_antismash)

    models = {}

    models['rf'] = supervised.rforest()
    models['ada_rf'] = supervised.adaRforest()
    models['knn'] = supervised.knn()
    models['nn'] = supervised.neural_network()

    for key, model in models.items():
        pass

if __name__ == "__main__":


    random.seed(1042)
    np.random.seed(1042)

    parser = argparse.ArgumentParser('bcg-finder')

    parser.add_argument('-t','--trainer', help='to train new NLP embedding')
    parser.add_argument('-a','--antismash', help='download sequences form antiSMASH DB')
    parser.add_argument('-m','--mibig', help='download sequences from MiBIG DB')
    parser.add_argument('-l','--learn', help='apply Dimensional Reduction on protein embedding')
    parser.add_argument('-b','--build', help='build clusters from genomes')
    parser.add_argument('-o','--operon', help='build clusters from genomes')
    parser.add_argument('-s','--supervised', help='Run Supervised learning on dataset')
    parser.add_argument('-u','--tuning', help='Hyperparameter tuning of models')
    parser.add_argument('-mp', '--mibig_predictions', help='Predict MiBiG classes')
    parser.add_argument('-p', '--predict', help='Predict class of of BGC')
    parser.add_argument('-i', '--split', help='Split dataset')
    parser.add_argument('-v', '--svd', help='Perform single value decomposition on matrix')
    parser.add_argument('-c', '--clustering', help='Cluster UMAP embedding')
    parser.add_argument('-d', '--distance', help='Build phylogenetic distance matrix')

    args = parser.parse_args()

    if args.trainer:
        if args.trainer.lower().endswith('gz'):
            print('--------------------------- START UniProt BioVec Training ----------------------------------------------')
            try:
                with gzip.open(os.path.join(os.getcwd(), 'bgc\\dataset\\{}'.format(args.trainer)), 'rb') as f_in:
                    with open(os.path.join(os.getcwd(), 'bgc\\dataset\\{}.fasta'.format(args.trainer.split('.')[0])), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except (gzip.BadGzipFile, FileNotFoundError) as e:
                print(e.__repr__())

            nlp = NLP(os.path.join(os.getcwd(), 'bgc\\dataset\\{}.fasta'.format(args.trainer.split('.')[0])))
            nlp.save('uniprot2vec')
            print('-------------------------------- ENDED SUCCESSFULLY -----------------------------------------------------')
        else:
            print('--------------------------- START UniProt Biovec Training -----------------------------------------------')
            nlp = NLP(os.path.join(os.getcwd(), 'bgc\\embedding\\{}.fasta'.format(args.trainer.split('.')[0])))
            nlp.save('uniprot2vec')
            print('-------------------------------- ENDED SUCCESSFULLY -----------------------------------------------------')

    elif args.mibig:
        pool = Pool()
        pool.map(mibig, os.listdir(os.path.join(os.getcwd(), 'bgc\\mibig\\nps')))
    elif args.antismash:
        pool = Pool()
        pool.map(antismash, os.listdir(os.path.join(os.getcwd(), 'bgc\\antismash\\antismash')))
    elif args.learn:

        # metrics = ['euclidean', 'manhattan', 'chebyshev']
        # metrics = [p for p in itertools.permutations(metrics, 2)]

        # for metric in metrics:
        #     learn(metric)

        learn(['euclidean', 'chebyshev'])

    elif args.build:
        pass
    elif args.operon:
        pass
    elif args.tuning:
        tuning()
    elif args.supervised:
        classify(10)
    elif args.split:
        split(args.split)
    else:
        # values = args.random.split(',')
        # rnd = RandomSequence(20000, 5000, 1000)
        # rnd.generatesequence()
        # rnd = RandomSequence(1500, 5000, 1700)
        # rnd.generatesequence()
        # rnd = RandomSequence(30000, 5000, 3500)
        # rnd.generatesequence()
        # rnd = RandomSequence(40000, 5000, 5000)
        # rnd.generatesequence()\

        pass