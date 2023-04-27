import os
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from scipy import sparse
from pybloom_live import BloomFilter
from Bio import pairwise2
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
from typing import Tuple

from clustcr import Clustering, datasets, metarepertoire


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


"""
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
"""


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


def visualize_features(features: pd.DataFrame):
    # Reference - https://svalkiers.github.io/clusTCR/docs/analyzing/features.html

    # Plot entropy of each cluster.
    fig, ax = plt.subplots()
    ax.bar(features.index, features['h'])
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy for Clusters')
    plt.show()

    # Plot the number of sequences in each cluster.
    fig, ax = plt.subplots()
    ax.bar(features.index, features['size'])
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Sequences')
    ax.set_title('Size of Clusters')
    plt.show()

    # Length of sequences in the cluster
    fig, ax = plt.subplots()
    ax.hist(features['length'], bins=range(25))
    ax.set_xlabel('CDR3 Sequence Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Length of CDR3 Sequences in Clusters')
    plt.show()

    # Plot the average and variance basicity of each cluster.
    fig, ax = plt.subplots()
    ax.scatter(features['basicity_avg'], features['basicity_var'])
    ax.set_xlabel('Average Basicity')
    ax.set_ylabel('Variance in Basicity')
    ax.set_title('Basicity in Clusters')
    plt.show()


def metareportoire(samples: list, training_sample_size: int) -> Tuple[pd.DataFrame, bool]:
    """
    Create a metareportoire from a list of samples.

    :param samples: A list of pandas DataFrame objects, each containing sequences to be concatenated.
    :param training_sample_size: An integer specifying the maximum number of sequences to reach.
    :return: A tuple containing the metareportoire as a pandas DataFrame and a boolean indicating whether the
            traning_sample_size was reached.
    """
    # Randomly select a sample.
    random.shuffle(samples)
    meta = samples[0]
    meta.drop_duplicates(inplace=True)

    # Concatenate the remaining samples until the training sample size is reached.
    for i in samples[1:]:
        meta = pd.concat([meta, i])
        meta.drop_duplicates(inplace=True)
        meta.reset_index()
        if len(meta) > training_sample_size:
            meta = meta.sample(training_sample_size)

    return meta, len(meta) >= training_sample_size


if __name__ == "__main__":
    persons = ['P1', 'P2', 'Q1', 'Q2', 'S1', 'S2']
    days = ['d0', 'd15']
    dict_df = read_files('data/Pogorelyy_YF/', persons, days, saveAll=False)
    dict_df = {key: value[:500] for key, value in dict_df.items()}

    # Calculate the total number of sequences in the dataset.
    total_sequences = sum(len(i) for i in dict_df.values())
    # Cluster size.
    faiss_cluster_size = 5000
    # Calculate recommended sample size.
    training_sample_size = round(1000 * (total_sequences / faiss_cluster_size))

    # TODO - filter out unresolved sequences??

    # Choose a random sample of sequences to create the metareportoire.
    samples = list(dict_df.values())
    # Keep calculating metareportoire with an increased cluster size until the sample size is reached.
    meta, success = metareportoire(samples, training_sample_size)
    while success is False:
        print(f'Metarepertoire: less sequences found than desired ({len(meta[0])} vs {training_sample_size}), \
                increasing faiss_cluster_size to {faiss_cluster_size * 1.1}')
        faiss_cluster_size *= 1.1
        training_sample_size = round(1000 * (total_sequences / faiss_cluster_size))
        meta, success = metareportoire(samples, training_sample_size)

    # Compute the maximum sequence length in the metareportoire.
    max_seq_len = meta.str.len().max()


    print(f'Perform clustering with a faiss_cluster_size of {faiss_cluster_size} and a training_sample_size of {training_sample_size} with a total of {total_sequences} sequences.')
    # Create clustering object.
    clustering = Clustering(faiss_training_data=meta, fitting_data_size=total_sequences, max_sequence_size=max_seq_len,
                            n_cpus='all', method='two-step', faiss_cluster_size=faiss_cluster_size, mcl_params=[1.2, 2])



    for i in dict_df.values():
        clustering.batch_precluster(i)
    for cluster in clustering.batch_cluster():
        print(cluster.clusters_df)

    clustering.batch_cleanup()

    # p1_d0 = df['P1_d0']
    #
    # # Reference - https://svalkiers.github.io/clusTCR/
    # clustering = Clustering(method='two-step', n_cpus='all', faiss_cluster_size=5000, mcl_params=[1.2, 2])
    #
    # results = clustering.fit(p1_d0[:1000])
    # # Include CDR3 alpha chain.
    #
    # # Retrieve dataframe containing clusters.
    # # results.clusters_df
    #
    # # Retrieve CDR3's in each cluster.
    # # print(results.cluster_contents())
    #
    # # Retrieve features of clustering.
    # features = results.compute_features(compute_pgen=True)
    #
    # visualize_features(features)
