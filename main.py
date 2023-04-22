import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
import hdbscan
from scipy import sparse
from pybloom_live import BloomFilter


def read_tsv(file_path: str, delimiter='\t'):
    """
    Reads in a CSV/TSV file and stores it in a pandas dataframe.

    Args:
        file_path (str): path to the CSV/TSV file.
        delimiter (str, optional): delimiter used in the file. Defaults to '\t'.

    Returns:
        pandas dataframe: the dataframe containing the data from the file.
    """
    try:
        return pd.read_csv(file_path, delimiter=delimiter)
    except Exception as e:
        print(f"Error reading file: {str(e)}")


def save_df(df: pd.DataFrame, file_path: str):
    """
    Saves a pandas dataframe to disk in pickle format.

    Args:
        df (pandas dataframe): the dataframe to save.
        file_path (str): path to the file to save the dataframe to.
    """
    df.to_pickle(file_path)


def read_files(path: str, persons: list[str], days: list[str], saveAll=True):
    """
    Reads in TSV files for given persons and days, and saves them to pickle format if they weren't already stored in pickles.

    Args:
        persons (list of str): list of person identifiers (e.g., ['P1', 'P2', 'P3']).
        days (list of str): list of day identifiers (e.g., ['d0', 'd15']).

    Returns:
        dict: dictionary of pandas dataframes, keyed by person-day (e.g., {'P1_d0': dataframe, 'P1_d15': dataframe, ...}).
    """
    dataframes = {}
    for person in persons:
        for day in days:
            filename = f"{person}_{day}.tsv"
            pickled_filename = f"{person}_{day}.pkl"
            if os.path.isfile(path + pickled_filename):
                # If pickled file exists, read from it
                dataframes[f"{person}_{day}"] = pd.read_pickle(path + pickled_filename) if saveAll else \
                    pd.read_pickle(path + pickled_filename)['CDR3.amino.acid.sequence']
                print(f"Read {path + pickled_filename} from disk.")
            else:
                # Otherwise, read TSV file and save to pickled file
                filepath = os.path.join(path, filename)
                df = read_tsv(filepath, delimiter='\t')
                save_df(df, path + pickled_filename)
                dataframes[f"{person}_{day}"] = df if saveAll else df['CDR3.amino.acid.sequence']
    return dataframes


if __name__ == "__main__":
    persons = ['P1', 'P2', 'Q1', 'Q2', 'S1', 'S2']
    days = ['d0', 'd15']
    df = read_files('data/Pogorelyy_YF/', persons, days, saveAll=False)['P1_d0']
    start = time.time()

    # # Choose the k-mer size.
    # k = 2
    # vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k))
    # X = vectorizer.fit_transform(df)
    #
    # # Convert X to a sparse matrix
    # X = X.tocsr()
    #
    # # Perform clustering
    # clustering = MiniBatchKMeans(n_clusters=100)
    # clusters = clustering.fit_predict(X)
    #
    # # Print out all clusters
    # for cluster_label in range(clustering.n_clusters):
    #     # Find the indices of TCR sequences belonging to this cluster
    #     cluster_indices = np.where(clusters == cluster_label)[0]
    #
    #     # Print out the TCR sequences belonging to this cluster
    #     print(f"Cluster {cluster_label}: {df.iloc[cluster_indices]}")

    # Convert TCR sequences to a Bloom filter representation with a k-mer size of 2
    bf = BloomFilter(capacity=len(df), error_rate=0.1)
    k = 2
    for seq in df:
        for i in range(len(seq) - k + 1):
            bf.add(seq[i:i + k])

    # Convert the Bloom filter to a sparse matrix representation
    indices = []
    indptr = [0]
    for i, seq in enumerate(df):
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            if kmer in bf:
                indices.append(hash(kmer) % len(bf))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=bool)
    X = sparse.csr_matrix((data, indices, indptr), shape=(len(df), len(bf)), dtype=bool)

    # Perform clustering
    clustering = MiniBatchKMeans(n_clusters=10)
    clusters = clustering.fit_predict(X)

    # Print out all clusters
    for cluster_label in range(clustering.n_clusters):
        # Find the indices of TCR sequences belonging to this cluster
        cluster_indices = np.where(clusters == cluster_label)[0]

        # Print out the TCR sequences belonging to this cluster
        print(f"Cluster {cluster_label}: {df.iloc[cluster_indices]}")

    print("Time taken: ", time.time() - start, " seconds")
