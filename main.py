import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from scipy import sparse
from pybloom_live import BloomFilter
from Bio import pairwise2
from Bio.Seq import Seq


def read_tsv(file_path: str, delimiter='\t') -> pd.DataFrame:
    """
    Reads in a CSV/TSV file and stores it in a pandas dataframe.

    :param file_path: str, path to the CSV/TSV file.
    :param delimiter: str, optional, delimiter used in the file. Defaults to '\t'.

    :return: pandas.DataFrame, the dataframe containing the data from the file.

    :raises: Exception if there's an error reading the file.
    """
    try:
        return pd.read_csv(file_path, delimiter=delimiter)
    except Exception as e:
        print(f"Error reading file: {str(e)}")


def save_df(df: pd.DataFrame, file_path: str):
    """
    Saves a pandas dataframe to disk in pickle format.

    :param df: pandas.DataFrame, the dataframe to save.
    :param file_path: str, path to the file to save the dataframe to.
    """
    df.to_pickle(file_path)


def read_files(path: str, persons: list[str], days: list[str], saveAll=True) -> dict:
    """
    Reads in TSV files for given persons and days, and saves them to pickle format if they weren't already stored in pickles.

    :param path: str, path to the directory containing the files.
    :param persons: list of str, list of person identifiers (e.g., ['P1', 'P2', 'P3']).
    :param days: list of str, list of day identifiers (e.g., ['d0', 'd15']).
    :param saveAll: bool, optional, whether to save all columns or just the CDR3 amino acid sequence. Defaults to True.

    :return: dict, dictionary of pandas dataframes, keyed by person-day (e.g., {'P1_d0': dataframe, 'P1_d15': dataframe, ...}).

    :raises: Exception if there's an error reading or saving a file.
    """
    dataframes = {}
    for person in persons:
        for day in days:
            filename = f"{person}_{day}.tsv"
            pickled_filename = f"{person}_{day}.pkl"
            if os.path.isfile(path + pickled_filename):
                # If pickled file exists, read from it.
                dataframes[f"{person}_{day}"] = pd.read_pickle(path + pickled_filename) if saveAll else \
                    pd.read_pickle(path + pickled_filename)['CDR3.amino.acid.sequence']
                print(f"Read {path + pickled_filename} from disk.")
            else:
                # Otherwise, read TSV file and save to pickled file.
                filepath = os.path.join(path, filename)
                df = read_tsv(filepath, delimiter='\t')
                save_df(df, path + pickled_filename)
                dataframes[f"{person}_{day}"] = df if saveAll else df['CDR3.amino.acid.sequence']
    return dataframes


def cluster_tcr_sequences(df: pd.DataFrame, k=2, error_rate=0.001, n_clusters=10) -> tuple[MiniBatchKMeans, np.ndarray]:
    """
    Performs clustering on a set of T-cell receptor (TCR) sequences using the Bloom filter and k-means clustering.

    :param df: a dataframe containing TCR sequences.
    :param k: the length of the k-mers to use for the Bloom filter. Defaults to 2.
    :param error_rate: the desired false positive rate for the Bloom filter. Defaults to 0.001.
    :param n_clusters: the desired number of clusters for the k-means clustering. Defaults to 10.

    :returns: a tuple containing the clustering object and the cluster assignments.

    :reference: https://llimllib.github.io/bloomfilter-tutorial/
    """
    # Bloom Filter is used to efficiently check whether an element is part of a set by some probability.
    # Set the bloom filter to store at least len(df) elements with a false positive rate
    # of error_rate, meaning that no more than error_rate are false positives.
    bf = BloomFilter(capacity=len(df), error_rate=error_rate)
    for seq in df:
        # Convert TCR sequences to a Bloom Filter representation with a k-mer size of k. This
        # creates a substring of length k from the TCR sequence. We use sliding window of k.
        for i in range(len(seq) - k + 1):
            bf.add(seq[i:i + k])

    # Store columns of indices of nonzero elements.
    indices = []
    # Store the pointer of each sequence of indices. (indicate end of rows)
    indices_ptr = [0]
    # Convert the Bloom filter to a sparse matrix representation with len(df) x len(bf).
    for seq in df:
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            # If kmer is in the Bloom Filter, the column index is added to 'indices'.
            if kmer in bf:
                indices.append(hash(kmer) % len(bf))
        indices_ptr.append(len(indices))

    # Store values of nonzero elements.
    data = np.ones(len(indices), dtype=bool)
    x = sparse.csr_matrix((data, indices, indices_ptr), shape=(len(df), len(bf)), dtype=bool)

    # Perform clustering.
    clustering = MiniBatchKMeans(n_clusters=n_clusters)
    return clustering, clustering.fit_predict(x)


def pairwise_sequence_identity(seq1, seq2):
    """
    Calculate the pairwise sequence identity between two T-cell receptor (TCR) sequences.

    :param seq1: The first TCR sequence as a string.
    :param seq2: The second TCR sequence as a string.

    :return: The pairwise sequence identity as a percentage (0-100) as a float.
    :raises ValueError: If input sequences are not of the same length.
    """
    # Convert input sequences to Seq objects.
    seq1_obj = Seq(seq1)
    seq2_obj = Seq(seq2)

    # Perform pairwise sequence alignment.
    alignments = pairwise2.align.globalxx(seq1_obj, seq2_obj)

    # Get the first alignment.
    alignment = alignments[0]

    # Calculate the pairwise sequence identity as a percentage
    return (alignment[2] / alignment[4]) * 100


if __name__ == "__main__":
    persons = ['P1', 'P2', 'Q1', 'Q2', 'S1', 'S2']
    days = ['d0', 'd15']
    df = read_files('data/Pogorelyy_YF/', persons, days, saveAll=False)

    p1_d0 = df['P1_d0']
    # Cluster TCR sequences.
    clustering1, clusters1 = cluster_tcr_sequences(p1_d0[:100000], k=3, error_rate=0.0001, n_clusters=10000)

    # for cluster_label in range(clustering1.n_clusters):
    #     # Retrieve indices of TCR sequences belonging to this cluster.
    #     cluster_indices = np.where(clusters1 == cluster_label)[0]
    #     print(f"Cluster {cluster_label}:")
    #     print(f"{p1_d0.iloc[cluster_indices]}")

    # Compute the pairwise sequence identities within each cluster.
    for cluster_label in range(clustering1.n_clusters):
        # Retrieve the TCR sequences belonging to this cluster.
        cluster_seqs = p1_d0.iloc[np.where(clusters1 == cluster_label)[0]].tolist()
        n_seqs = len(cluster_seqs)
        # Compute the pairwise sequence identities.
        sequence_identities = []
        for i, seq1 in enumerate(cluster_seqs):
            for j, seq2 in enumerate(cluster_seqs[i + 1:], i + 1):
                identity = pairwise_sequence_identity(seq1, seq2)
                sequence_identities.append(identity)
        # Compute mean and standard deviation of pairwise sequence identities.
        mean_identity = np.mean(sequence_identities)
        std_dev = np.std(sequence_identities)
        min_identity = min(sequence_identities, default=np.nan)
        max_identity = max(sequence_identities, default=np.nan)
        print(f"Cluster {cluster_label} (n={n_seqs}): Mean identity = {mean_identity:.3f}%, Std dev = {std_dev:.3f}%")
        if sequence_identities:
            print(
                f"    - This cluster contains TCR sequences that {'share' if mean_identity > 50 else 'do not share'} a moderate level of sequence similarity, with an average pairwise identity of {mean_identity:.1f}% and a range of identities from {min_identity:.1f}% to {max_identity:.1f}%.")
        else:
            print("    - This cluster contains no TCR sequences.")
