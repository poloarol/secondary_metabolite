""" app.py """

import os, gzip, time, joblib, shutil, json, biovec
import random, argparse, itertools, statistics, itertools

from typing import Any, Dict, List, Tuple
from multiprocessing import Pool
from numpy.lib import stride_tricks

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from Bio import Entrez, SeqIO

# from pinky.smiles import smilin
# from pinky.fingerprints import ecfp

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from utilities.reader import ReadGB
from utilities.download import Scraper

from utilities.analysis import NLP
from utilities.analysis import Clustering
from utilities.analysis import Supervised
from utilities.analysis import DimensionalReduction

from chemspipy import ChemSpider

import pubchempy
from pubchempy import Compound

Entrez.email = "adjon081@uottawa.ca"


def load_files():
    
    """ 
    Load files files to the workspace.

    Return
    ------

    seed (int): Training seed for ML
    train (pd.DataFrame): training set
    test (pd.DataFrame): testing set

    """
    
    # seed, train = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\train.csv'))
    # seed, test = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\antismash\\test.csv'))

    seed, data = shuffle(os.path.join(os.getcwd(), 'bgc\\dataset\\mibig\\mibig.csv'))

    scaler = RobustScaler().fit(data.iloc[:, 1:])

    train = np.column_stack((data.iloc[:, 0].astype(int), scaler.transform(data.iloc[:, 1:])))
    train = pd.DataFrame(train)
    
    return seed, train

def umap_learn(metric):
    """

        Optimize UMAP by varying the metrics and neighbours. 
        The UMAP optimization is tested using supervised learning.
        The optmization results are written to a file, with each containing
            accuracy, balanced accuracy, Cohen's Kappa and Matt. Corr. Coef

        Parameters
        ----------
        metric (tuple[str, str]) e.g. (euclidean, euclidean)
    """

    seed, train = load_files()

    dim_reduction = DimensionalReduction(seed, train)

    neighbours: List = [15, 30, 45, 60, 75, 90]
    distances: List = [0.1, 0.2, 0.3, 0.4, 0.5]
    weight: List = [0.1, 0.2, 0.3, 0.4, 0.5]

    tmp = [neighbours, distances, weight]

    all_permutations: List = list(itertools.product(*tmp))

    col_names: List = ['Model', 'Neighbours', 'Distance', 'Weight', 'Trustworthiness', 'Accuracy', 'Bal. Accuracy', 'Cohen K', 'Matt. Corr. Coef']
    analysis_data: pd.DataFrame = pd.DataFrame(columns=col_names)
    

    for permutation in all_permutations:

        print('UMAP Evaluation - num. of neighbours: {} & Min. Dist: {} & Weight: {}'.format(permutation[0], permutation[1], permutation[2]))

        rb = RobustScaler()
        rb.fit(train.iloc[:, 1:])

        with open(os.path.join(os.getcwd(), 'bgc\\models\\scaler\\RobustScaler.model'), 'wb') as f:
            joblib.dump(rb, f)

        umap = dim_reduction.umap_learn(in_metric=metric[0], out_metric=metric[1], neighbours=permutation[0], distance=permutation[1], weight=permutation[2])
        transformed_train = pd.DataFrame(np.column_stack((train.iloc[:, 0], umap.transform(rb.transform(train.iloc[:, 1:])))))

        a = train.iloc[:, 1:].to_numpy()
        b = transformed_train.iloc[:, 1:].to_numpy()

        trust = dim_reduction.trustworthiness(a, b, permutation[0], metric[1])
            
        a, b = train_test_split(transformed_train, test_size=0.33, random_state=seed)

        try:

            evaluate_model(seed=seed, train=a, test=b, trust=trust, df=analysis_data, params=permutation)

        except Exception as e:

            print(e.__repr__())
    
    analysis_data.to_excel('{}_{}.xlsx'.format(metric[0], metric[1]), index=False)

def lda_learn(n_component: int):
    
    """
    Optimize LDA algorithm to determine the number of components which 
    allow for efficient seperation.
    """
    
    seed, train, test = load_files()

    dim_reduction = DimensionalReduction(seed, train)
    
    lda = dim_reduction.lda_learn(components=n_component)

    
    transformed_train = np.column_stack((train.iloc[:, 0], lda.transform(train.iloc[:, 1:])))
    transformed_test = np.column_stack((test.iloc[:, 0], lda.transform(test.iloc[:, 1:])))
    
    transformed_train = pd.DataFrame(transformed_train)
    transformed_test = pd.DataFrame(transformed_test)
    
    print('LDA Evaluation begins')
    
    evaluate_model(seed=seed, train=transformed_train, test=transformed_test)
    
    print('LDA Evaluation ends')

def nca_learn(n_component: int):
    
    """
    Optimize NCA algorithm to determine the number of components which 
    allow for efficient seperation.
    """
    
    seed, train, test = load_files()

    dim_reduction = DimensionalReduction(seed, train)
    
    nca = dim_reduction.nca_learn(components=n_component)
    
    transformed_train = np.column_stack((train.iloc[:, 0], nca.transform(train.iloc[:, 1:])))
    transformed_test = np.column_stack((test.iloc[:, 0], nca.transform(test.iloc[:, 1:])))
    
    transformed_train = pd.DataFrame(transformed_train)
    transformed_test = pd.DataFrame(transformed_test)
    
    print('NCA Evaluation begins')
    
    evaluate_model(seed=seed, train=transformed_train, test=transformed_test)
    
    print('NCA Evaluation ends')

def evaluate_model(seed: int, train: pd.DataFrame, test: pd.DataFrame, trust: float, df: pd.DataFrame, params: Tuple) -> None:

    supervised = Supervised(seed, train, test)

    models = {}

    models['rf'] = supervised.rforest()
    models['ada_rf'] = supervised.adaRforest()
    models['nn'] = supervised.neural_network()
    models['knn'] = supervised.knn()
    models['xg'] = supervised.xgboost()

    for key, model in models.items():
        accuracy, bal_accuracy, cohen_kappa, matt_corr_coef = supervised.evaluate_model(model, 3, 10)

        acc: int = statistics.mean(accuracy)
        bal_acc: int = statistics.mean(bal_accuracy)
        c_kappa: int = statistics.mean(cohen_kappa)
        matt_corr: int = statistics.mean(matt_corr_coef)
        
        new_row = [key, params[0], params[1], params[2], trust, acc, bal_acc, c_kappa, matt_corr]

        df.loc[len(df.index)] = new_row

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

def tuning(neighbour: int = 90, input_metric: str = 'chebyshev', output_metric: str = 'chebyshev', seed: int = 1042):  

    seed, train = load_files()

    rb: Any = None

    dim_reduction = DimensionalReduction(seed, train)
    umap = dim_reduction.umap_learn(in_metric="chebyshev", out_metric='euclidean', neighbours=30, distance=0.1, weight=0.3)

    with open(os.path.join(os.getcwd(), 'bgc\\models\\scaler\\RobustScaler.model'), 'rb') as f:
        rb = joblib.load(f)
    
    with open(os.path.join(os.getcwd(), 'bgc\\models\\umap\\umap_chebyshev_euclidean_30_0.1_0.3.model'), 'rb') as f:
        umap = joblib.load(f)
    
    transformed_train = pd.DataFrame(train.iloc[:, 0], umap.transform(train.iloc[:, 1:]))
    tmp_train, tmp_test = train_test_split(transformed_train, test_size=0.33, random_state=seed)

    supervised = Supervised(seed, tmp_train, tmp_test)
    supervised.hyperparameter_tuning()

def shuffle(filename: str):
    """ shuffle the pandas dataset and divide it into a 4 (train) : 1 (test) ratio. """

    seed: int = 42
    data = pd.read_csv(filename, low_memory=True, header=None)
    data = clean_dataset(data)
    data = data.sample(frac=1).reset_index(drop=True)
    return seed, data

def clean_dataset(df):
    """ Cleans dataset and removes redundant information """
    print('start - cleaning dataset')
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    print('done - cleaning dataset')
    return df[indices_to_keep].astype(np.float64)

def mibig(file):
    """ access the mibig database, download genbank files and convert them to vectors"""
    
    data = np.array([])

    if os.path.isfile(os.path.join(os.getcwd(), 'bgc\\mibig\\nps\\align\\{0}'.format(file))):
        with open(os.path.join(os.getcwd(), 'bgc\\mibig\\nps\\align\\{0}'.format(file))) as lines:
            for line in lines:
                scraper = Scraper(line.strip())
                scraper.mibig_download()
                path = os.path.join(os.getcwd(), 'tmp\\gbk\\{}.gbk'.format(line.strip()))
                rb = ReadGB(path)
                rb.to_fasta(os.path.join(os.getcwd(), 'bgc\\mibig\\bgcs\\{}.fasta'.format(line.strip())), line.strip())
                vector = rb.get_vector()
                vector = vector.ravel()

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
                vector = vector.ravel()

                if vector.shape[0] == 300:
                    if data.size == 0:
                        data = np.array([vector])
                    elif vector.size != 0:
                        data = np.concatenate((data, np.array([vector])))
    np.savetxt('bgc\\antismash\\vectors\\{}_uniprot2vec.csv'.format(file.split('.')[0]), data, delimiter=',')

def get_best_umap_models():

    """
        Generate final models and save them.
    """

    seed, train = load_files()

    dim_reduction = DimensionalReduction(seed, train)

    all_parameters: Dict = {
        'chebyshev-chebyshev': [[75, 0.1, 0.1], [60, 0.1, 0.2], [45, 0.1, 0.2], [30, 0.1, 0.2], [45, 0.2, 0.2], [45, 0.2, 0.3], [60, 0.2, 0.2]],
        'chebyshev-manhattan': [[15, 0.1, 0.1], [15, 0.3, 0.2], [15, 0.2, 0.1]],
        'chebyshev-euclidean': [[30, 0.1, 0.2], [30, 0.1, 0.3], [30, 0.2, 0.3], [30, 0.4, 0.3]],
        'euclidean-chebyshev': [[15, 0.4, 0.4], [15, 0.1, 0.3], [15, 0.5, 0.3]]
    }

    df = pd.DataFrame(columns=['Model', 'In-metric', 'Out-metric', 'Neigh.', 'Dist.', 'Weight', 'trustworthiness', 'acc', 'bal. acc', 'kappa', 'matt'])

    for key, values in all_parameters.items():
        for value in values:
            metric = key.split('-')
            umap = dim_reduction.umap_learn(in_metric='chebyshev', out_metric=metric[1], components=2, neighbours=value[0], distance=value[1], weight=value[2])

            transformed_train = pd.DataFrame(np.column_stack((train.iloc[:, 0], umap.transform(train.iloc[:, 1:]))), columns=['biosyn_class', 'Dim1', 'Dim2'])
            transformed_train.to_excel('embedding_{}_{}_{}_{}.xlsx'.format(key, value[0], value[1], value[2]), index=False)

            a = train.iloc[:, 1:].to_numpy()
            b = transformed_train.iloc[:, 1:].to_numpy()

            trust = round(dim_reduction.trustworthiness(a, b, value[0], metric[1]), 4)

            tmp_train, tmp_test = train_test_split(transformed_train, test_size=0.33, random_state=seed)

            supervised = Supervised(seed, tmp_train, tmp_test)

            models = {}

            with open(os.path.join(
                        os.getcwd(), 
                        'bgc\\models\\umap\\umap_{}_{}_{}_{}_{}.model'.format(metric[0], metric[1], value[0], value[1], value[2])), 'wb') as f:
                joblib.dump(umap, f)

            models['rf'] = supervised.rforest()
            models['ada_rf'] = supervised.adaRforest()
            models['nn'] = supervised.neural_network()
            models['knn'] = supervised.knn()
            models['xg'] = supervised.xgboost()

            for k_model, model in models.items():
                accuracy, bal_accuracy, cohen_kappa, matt_corr_coef = supervised.evaluate_model(model, 3, 10)

                acc: str = '{} +/- {}'.format(round(statistics.mean(accuracy), 5), round(statistics.stdev(accuracy), 5))
                bal_acc: str = '{} +/- {}'.format(round(statistics.mean(bal_accuracy), 5), round(statistics.stdev(bal_accuracy), 5))
                c_kappa: str = '{} +/- {}'.format(round(statistics.mean(cohen_kappa), 5), round(statistics.stdev(cohen_kappa), 5))
                matt_corr: str = '{} +/- {}'.format(round(statistics.mean(matt_corr_coef), 5), round(statistics.stdev(matt_corr_coef), 5))

                tmp = value

                row = [k_model, metric[0], metric[1], tmp[0], tmp[1], tmp[2], trust, acc, bal_acc, c_kappa, matt_corr]

                df.loc[len(df.index)] = row
    
    df.to_excel('umap_best_models_params.xlsx', index=False)

def __parse_json__(file):

    row = []

    cs = ChemSpider("rJUZR1hq5XdRErFszO7NfZPv1GwBXzOO")

    try:
        with open(os.path.join(os.getcwd(), 'bgc\\mibig\\mibig_json\\{}'.format(file))) as json_file:
            tmp = json_file.read()
            bgc = json.loads(tmp)

            for key, value in bgc.items():
                if isinstance(value, dict):
                    tmp: str = 'None'
                    for k, cpds in value.items():
                        if k == 'biosyn_class':
                            tmp = cpds[0]
                        if k == 'compounds':
                            for cpd in cpds:
                                formulae = None
                                if 'database_deposited' in cpd and cpd['database_deposited'] == True:
                                    db: str = cpd['databases_deposited'][0]
                                    db_id = 'None'
                                    if 'pubchem_id' in cpd:
                                        db_id = cpd['pubchem_id']
                                        try:
                                            c1 = Compound.from_cid(db_id)
                                            formulae = c1.canonical_smiles
                                        except Exception as e:
                                            print(db_id, e.__repr__())
                                    elif 'chemspider_id' in cpd:
                                        db_id = cpd['chemspider_id']
                                        try:
                                            c1 = cs.get_compound(db_id)
                                            formulae = c1.smiles
                                        except Exception as e:
                                            print(db_id, e.__repr__())
                                    elif 'chembl_id' in cpd:
                                        try:
                                            c1 = Compound.from_cid(db_id)
                                            formulae = c1.canonical_smiles
                                        except Exception as e:
                                            print(db_id, e.__repr__())
                                    else:
                                        db_id = 'None'
                                    compound: str = cpd['compound'] if 'compound' in cpd else 'None'

                                    row = [db, db_id, tmp, compound, formulae]
                                    time.sleep(5)
                                    print(row)

        return row

    except Exception as e:
        print(e.__repr__())
    
def parse_json():

    """
    Parse all BGC JSON files to extract db, db_id, compound, structure
    remove all duplicates and write to file
    """

    col_names: List = ['database', 'database_id', 'biosyn_class','compound', 'smiles']
    
    df: pd.DataFrame = pd.DataFrame(columns=col_names)

    for file in os.listdir(os.path.join(os.getcwd(), 'bgc\\mibig\\mibig_json')):
        tmp = __parse_json__(file)

        if tmp:
            df.loc[len(df.index)] = tmp

    df = df.drop_duplicates(subset=['database', 'database_id', 'compound'], keep='last')
    df.to_csv('mibig_parsed_json.csv', index=False)

def cluster():

    data = pd.read_csv(os.path.join(os.getcwd(), 'subset_mibig_tanimoto.csv'))
    tmp = pd.DataFrame(data.iloc[:, :236])

    rb = RobustScaler()
    rb.fit(tmp)

    clustering = Clustering(rb.transform(tmp))
    df = pd.DataFrame()

    linkages = ['ward', 'complete', 'average', 'single']
    for linkage in linkages:
        model = clustering.hierachical_clustering(linkage=linkage)
        df[linkage] = model.labels_

        link_matrix = plot_dendrogram(model, truncate_mode='level', p=3)

        link_df = pd.DataFrame(link_matrix)
        link_df['biosyn_class'] = data['biosyn_class']

        link_df.to_csv('linkage_matrix_dendrogram_{}.csv'.format(linkage), index=False)


    
    df['true_label'] = data['biosyn_class'].to_list()
    
    df.to_csv('predicted_labels.csv', index=False)

def plot_dendrogram(model, **kwargs):

    """ Create linkage matrix and the plot the dendrogram """

    # create the counts of samples under each node
    counts: int = np.zeros(model.children_.shape[0])
    n_samples: int = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count: int = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
    
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    return linkage_matrix

    # graph = dendrogram(linkage_matrix, **kwargs)

    # return graph

def tanimoto_clustering():

    """
    Calculate pairwise tanimoto score between all molecule within MiBIG DB
    """

    mibig = pd.read_csv(os.path.join(os.getcwd(), 'subset_mibig_parsed_json.csv'))
    smiles = mibig['smiles'].to_list()

    scores = pd.DataFrame(columns=mibig['compound'])

    for first in smiles:
        tmp = []
        try:
            morgan_fpt1 = list(ecfp(smilin(first), radius=2))
        except Exception as e:
            print(e.__repr__())
        for second in smiles:
            try:
                morgan_fpt2 = list(ecfp(smilin(second), radius=2))

                numerator = list(set(morgan_fpt1) & set(morgan_fpt1))
                denominator = list(set(morgan_fpt1) | set(morgan_fpt2))

                tmp.append(len(numerator)/len(denominator))

            except Exception as e:
                tmp.append(0)
                print(e.__repr__())  
        
        scores.loc[len(scores.index)] = tmp
    
    scores['biosyn_class'] = mibig['biosyn_class'].to_list()
    scores['label'] = mibig['compound'].to_list()

    scores.to_csv(os.path.join(os.getcwd(), 'subset_mibig_tanimoto.csv'), index=False)

def download_sequences(accessions: List):

    sequences: List = []
    model =  biovec.models.load_protvec(os.path.join(os.getcwd(), 'bgc\\models\\biovec\\uniprot2vec.model'))

    for accession in accessions:

        try:
            handle = Entrez.efetch(db="protein", id=accession, rettype='fasta', retmode='text')
            for record in SeqIO.parse(handle, 'fasta'):
                tmp = np.array(model.to_vecs(record.seq))
                tmp = tmp.ravel()
                sequences.append(tmp)
        except Exception as e:
            print(accession, e.__repr__())
    
    return sequences

def cosine_similarity():

    """
    Calculate the cosine similary between a two proteins.
    These pair of proteins are homologous.
    Uses biovec for the convertion of sequences to vectors.
    """

    data = pd.read_csv(os.path.join(os.getcwd(), 'bgc\\dataset\\housekeeping.csv'))
    data = data.iloc[:, 1:]

    col_names: List = ['E. coli', 'Pseudomonas aeruginosa', 'Streptomyces sp', 'Saccharomyces cerevisiae', 'Mus musculus']
    cosine_similarity_df = pd.DataFrame(columns=col_names)

    for index, row in data.iterrows():
        row = row.to_list()
        vectors = download_sequences(row)

        tmp = []

        for vector in vectors[:-1]:
            numerator = np.dot(vectors[-1], vector)
            denominator = np.linalg.norm(vectors[-1]) * np.linalg.norm(vector)

            cos_sim = numerator/denominator
            tmp.append(cos_sim)
        
        cosine_similarity_df.loc[len(cosine_similarity_df.index)] = tmp
    
    cosine_similarity_df.to_csv('cosine_similarity.csv', index=False)

if __name__ == "__main__":


    random.seed(1042)
    np.random.seed(1042)

    parser = argparse.ArgumentParser('bcg-finder')

    parser.add_argument('-e', '--embedding', help='to train new NLP embedding')
    parser.add_argument('-a', '--antismash', help='download sequences form antiSMASH DB')
    parser.add_argument('-m', '--mibig', help='download sequences from MiBIG DB')
    parser.add_argument('-y', '--hyperparameter_tuning', help='Hyperparameter tuning of models')
    parser.add_argument('-i', '--split', help='Split dataset')
    parser.add_argument('-l', '--lda', help='Evaluate LDA DR algorithm')
    parser.add_argument('-n', '--nca', help='Evaluate NCA DR algorithm')
    parser.add_argument('-u', '--umap', help='Evaluate UMAP DR algorithm')
    parser.add_argument('-t', '--tanimoto', help='Build distance matrix of molecules using Tanimoto scores')
    parser.add_argument('-c', '--cosine', help='Calculates cosine similarity between a human proteins and designated homologues')
    parser.add_argument('-g', '--clustering', help="Cluster Tanimoto Scores")
    parser.add_argument('-bu', '--best_umap', help='Build best UMAP models and save them')

    args = parser.parse_args()

    if args.embedding:
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
        pool.map(mibig, os.listdir(os.path.join(os.getcwd(), 'bgc\\mibig\\nps\\align')))
    elif args.antismash:
        pool = Pool()
        pool.map(antismash, os.listdir(os.path.join(os.getcwd(), 'bgc\\antismash\\antismash')))
    elif args.umap:

        metrics = ['euclidean', 'manhattan', 'chebyshev']
        metrics = [p for p in itertools.product(metrics, repeat=2)]

        for metric in metrics:
            umap_learn(metric)
    
    elif args.lda:     
        components: int = 4
        for i in range(1, components):
            lda_learn(i)
    elif args.nca:
        components: int = 4
        for i in range(1, components):
            nca_learn(i)
    elif args.hyperparameter_tuning:
        tuning()
    elif args.split:
        split(args.split)
    elif args.tanimoto:
        tanimoto_clustering()
    elif args.cosine:
        cosine_similarity()
    elif args.clustering:
        cluster()
    elif args.best_umap:
        get_best_umap_models()
    else:
        try:
            parse_json()
        except Exception as e:
            print(e.__repr__())
        # values = args.random.split(',')
        # rnd = RandomSequence(20000, 5000, 1000)
        # rnd.generatesequence()
        # rnd = RandomSequence(1500, 5000, 1700)
        # rnd.generatesequence()
        # rnd = RandomSequence(30000, 5000, 3500)
        # rnd.generatesequence()
        # rnd = RandomSequence(40000, 5000, 5000)
        # rnd.generatesequence()