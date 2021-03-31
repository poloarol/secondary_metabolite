""" app.py """

import os
import gzip
from os import path
import queue
import time
from typing import List
import joblib
import shutil
import argparse
import itertools
import statistics

from multiprocessing import Pool

import biovec as bv
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np

from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator

from matplotlib import pyplot as plt

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utilities.download import Scraper
from utilities.reader import ReadGB
from utilities.analysis import NLP
from utilities.analysis import Supervised
from utilities.analysis import DimensionalReduction


def svd(filename: str):
    data = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\mibig\\{0}'.format(filename)))

    unsupervised = DimensionalReduction(data[0], data[1])

    pcs = unsupervised.scree_plot()
    num_vars = np.arange(data[1].iloc[:, 1:].shape[1]) + 1
    pcs = np.column_stack((num_vars, pcs))

    np.savetxt('bgc\\embedding\\pcs.csv', pcs, delimiter=',')

def learn(filename: str, dim: int):
    """
    """

    print('--------------------------- Processing data for UMAP begins  ----------------------------------------------')
    data = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\mibig\\{0}'.format(filename)))

    print('--------------------------------- Processing data ends -----------------------------------------------------')
    
    scaler = RobustScaler()
    scaled_data = np.column_stack((data[1].iloc[:, 0], scaler.fit_transform(data[1].iloc[:, 1:])))

    scaled_data = pd.DataFrame(scaled_data)

    dim_reduction = DimensionalReduction(data[0], scaled_data)

    print('---------------------------------- UMAP learning begins ------------------------------------------------')

    metrics = ['euclidean', 'chebyshev', 'manhattan', 'mahalanobis', 'minkowski', 'canberra']
    weights = [0, 0.001, 0.01, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    distances = [0, 0.001, 0.01, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    neighbours = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90, 100]

    comb = [distances, neighbours, weights]
    comb = [p for p in itertools.product(*comb)]

    metrics = [p for p in itertools.permutations(metrics, 2)]

    models = {}

    with open(os.path.join(os.getcwd(), 'bgc\\embedding\\umap_best_params.txt'), 'w') as f:

        for metric in metrics:
            for c in comb:

                tmp = dim_reduction.umap(min_dist=c[2], neighbours=c[1], output_metric=metric[0], weight=c[0], input_metric=metric[1])
                test = dim_reduction.test_embedding(tmp[0], 'umap_test_euclidean')

                try:
                    supervised = Supervised(data[0], tmp[1], test)
                    
                    models['rf'] = supervised.rforest()
                    models['ada_rf'] = supervised.adaRforest()
                    models['xgboost'] = supervised.xgboost()
                    models['knn'] = supervised.knn()
                    models['nn'] = supervised.neural_network()

                    for key, model in models.items():
                        scores = supervised.evaluate_model(model, 5, 3)

                        # input_metric, output_metric, min_dist, weight, model, av. accuracy

                        if statistics.mean(scores) >= 0.7:
                            s = '{}, {}, {}, {}, {},  {}, {}\n'.format(metric[0], metric[1], c[1], c[0], c[2], key, statistics.mean(scores))
                            f.write(s)
                except Exception as e:
                    print(e.__repr__())

            

    print('------------------------------ UMAP learning ends ------------------------------------------------')

    # joblib.dump(scaler, os.path.join(os.getcwd(), 'bgc\\models\\scaler\\RobustScaler'))

def split(filename: str):
    data = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\{0}'.format(filename)))
    tmp = data[1].sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(tmp, test_size=0.25, random_state=data[0])

    np.savetxt(os.path.join(os.getcwd(), 'bgc\\dataset\\mibig\\train.csv'), train, comments='', delimiter=',')
    np.savetxt(os.path.join(os.getcwd(), 'bgc\\dataset\\mibig\\validation.csv'), test, comments='', delimiter=',')

def classify(dim: int = 3):

    train = shuffle(os.path.join(os.getcwd(), 'bgc\\embedding\\umap_train_euclidean_minkowski.csv'))
    test = shuffle(os.path.join(os.getcwd(), 'bgc\\embedding\\umap_test_euclidean_minkowski.csv'))

    models = {}

    supervised = Supervised(train[0], train[1], test[1])

    models['rf_{0}'.format(dim)] = supervised.rforest()
    models['ada_rf_{0}'.format(dim)] = supervised.adaRforest()
    models['xgboost_{0}'.format(dim)] = supervised.xgboost()
    models['knn_{0}'.format(dim)] = supervised.knn()
    # models['nn_{0}'.format(dim)] = supervised.neural_network()

    for key, model in models.items():
        scores = supervised.evaluate_model(model, 5, 3)

        print('{}: {}'.format(key, scores))
    
def tuning(filename: str):

    data = shuffle(os.path.join(os.getcwd(), 'bgc\\embedding\\{0}'.format(filename)))
    
    supervised = Supervised(data[0], data[1])
    supervised.hyperparameter_tuning()

def shuffle(filename: str):
    """ shuffle the pandas dataset and divide it into a 4 (train) : 1 (test) ratio. """

    seed: int = 1042
    data = pd.read_csv(filename, low_memory=True, header=None)
    data = clean_dataset(data)
    data = data.sample(frac=1).reset_index(drop=True)
    return (seed, data)

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

def mibig_predictions(dim: int):
    
    data = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\mibig\\validation.csv'))

    rob_scaler = joblib.load(os.path.join(os.getcwd(), 'bgc\\models\\scaler\\RobustScaler_{0}.model'.format(dim)))
    # pca = joblib.load( os.path.join(os.getcwd(), 'bgc\\models\\unsupervised\\pca_10.model'))
    umap = joblib.load(os.path.join(os.getcwd(), 'bgc\\models\\unsupervised\\umap_chebyshev_manhattan_train_{0}.model'.format(dim)))

    tmp = rob_scaler.transform(data[1].iloc[:, 1:])
    tmp = umap.transform(tmp)

    tmp = pd.DataFrame(np.column_stack((data[1].iloc[:, 0], tmp)))

    # std_scaler = joblib.load(os.path.join(os.getcwd(), 'bgc\\models\\scaler\\StadardScaler_{0}.model'.format(dim)))
    # scaled_data = std_scaler.transform(tmp)

    models = {}

    # models['rf_{0}'.format(dim)] = joblib.load(os.path.join(os.getcwd(), 'bgc\\models\\supervised\\rf_{0}.model'.format(dim)))
    # models['knn_{0}hp'.format(dim)] = joblib.load(os.path.join(os.getcwd(), 'bgc\\models\\supervised\\knn_{0}hp.model'.format(dim)))
    # models['nn_{0}hp'.format(dim)] = joblib.load(os.path.join(os.getcwd(), 'bgc\\models\\supervised\\nn_{0}hp.model'.format(dim)))

    print('===========================================================================')

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

def distance_matrix(file):

    alignments = AlignIO.read(file, 'fasta')
    # calculator = DistanceCalculator(aln, )

    for alignment in alignments:
        if 'BGC' in alignment.id:
            print(alignment.id)

if __name__ == "__main__":


    dims = [2]

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
    np.random.seed(1042)

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
        for dim in dims:
            learn(args.learn, dim)
    elif args.build:
        pass
    elif args.operon:
        pass
    elif args.tuning:
        for dim in dims:
            tuning('umap_euclidean_euclidean_train_{0}.csv'.format(dim))
    elif args.supervised:
        classify(10)
    elif args.mibig_predictions:
        for dim in dims:
            mibig_predictions(dim)
    elif args.split:
        split(args.split)
    elif args.svd:
        svd(args.svd)
    elif args.distance:
        distance_matrix(os.path.join(os.getcwd(), 'bgc\\msa\\{0}'.format(args.distance)))
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