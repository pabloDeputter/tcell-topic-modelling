import os
import re
import random
import time
import shutil
import warnings

import pandas as pd
import numpy as np

from clustcr.clustering.clustering import ClusteringResult
from Bio import BiopythonDeprecationWarning
from typing import Tuple
from clustcr import Clustering

import utils


def metareportoire(samples: list, training_sample_size: int) -> Tuple[pd.DataFrame, bool]:
    """
    Create a metareportoire from a list of samples.

    :param samples: A list of pandas DataFrame objects, each containing sequences to be concatenated.
    :param training_sample_size: An integer specifying the maximum number of sequences to reach.
    :return: A tuple containing the metareportoire as a pandas DataFrame and a boolean indicating whether the
            training_sample_size was reached.
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
            return meta.sample(training_sample_size), True

    return meta, False


def preprocess_tcr_sequences(df: pd.DataFrame, unresolved: str = 'X', min_read_count: int = 1) -> pd.DataFrame:
    """
    Preprocesses TCR sequences in a DataFrame by removing sequences containing unresolved amino acids, removing duplicates,
    filtering out sequences with a low read count, and selecting a maximum number of sequences.

    :param df: pd.DataFrame, DataFrame containing TCR sequences.
    :param unresolved: str, optional, string indicating unresolved amino acids. Defaults to 'X'.
    :param min_read_count: int, optional, minimum read count threshold. Sequences with read count below this value will be removed. Defaults to 1.

    :return: pd.DataFrame, preprocessed DataFrame containing TCR sequences.
    """
    # Remove sequences containing unresolved amino acids
    df_filtered = df[~df['CDR3.amino.acid.sequence'].str.contains(unresolved)]
    # Remove duplicates
    df_filtered.drop_duplicates(subset='CDR3.amino.acid.sequence', inplace=True)
    # Remove sequences with a low read count
    return df_filtered.loc[df_filtered['Read.count'] >= min_read_count]


def contains_stop_codon(s: str) -> bool:
    """
    Check whether a given DNA sequence contains any stop codons.

    :param s: A string representing a DNA sequence.
    :return: A boolean value indicating whether the input DNA sequence contains a stop codon.
             Returns True if a stop codon is found, False otherwise.
    """
    stop_codons = ['TAA', 'TAG', 'TGA']
    return any(s[i:i + 3] in stop_codons for i in range(0, len(s) - 2, 3))


def contains_invalid_chars(seq: str) -> bool:
    """
    Check if a given CDR3 amino acid sequence contains any invalid characters.

    :param seq: A string representing a CDR3 amino acid sequence.
    :return: A boolean value indicating whether the input sequence contains any invalid characters.
             Returns True if invalid characters are found, False otherwise.
    """
    valid_chars_regex = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
    return not bool(valid_chars_regex.match(seq))


def cluster_chunks(data: pd.DataFrame, n_cpus: str = '8'):
    """
    Performs clustering on a large dataset of CDR3 amino acid sequences by dividing it into smaller
    chunks and clustering each chunk independently.

    :param n_cpus: The number of CPUs to use for clustering.
    :param data: A pandas DataFrame containing the CDR3 amino acid sequences to cluster.

    :return: A list of Cluster objects containing the results of the clustering for each chunk.
    """
    # Calculate the total number of sequences in the dataset.
    total_sequences = data.shape[0]
    # Cluster size.
    faiss_cluster_size = 5000
    # Calculate recommended sample size.
    training_sample_size = round(1000 * (total_sequences / faiss_cluster_size))

    # Create metareportoire.
    meta = data[['CDR3.amino.acid.sequence']].copy()
    # Remove duplicates.
    meta.drop_duplicates(inplace=True)
    # Randomly sample the metareportoire.
    if len(meta) > training_sample_size:
        meta = meta.sample(training_sample_size)
    # Compute the maximum sequence length in the metareportoire.
    max_seq_len = meta['CDR3.amino.acid.sequence'].str.len().max()

    # Create a dictionary for chunks.
    if not os.path.exists('data/Pogorelyy_YF/chunks'):
        os.makedirs('data/Pogorelyy_YF/chunks')
    # Clear .pkl files in 'chunks' directory.
    [os.remove(os.path.join(dirpath, file)) for dirpath, dirnames, filenames in os.walk('data/Pogorelyy_YF/chunks') for
     file
     in filenames if file.endswith('.pkl')
     ]
    time.sleep(1)

    # Clear cluster_batch directories.
    for dir in os.listdir('./'):
        if dir.startswith('clustcr_batch') and os.path.isdir(os.path.join('./', dir)):
            shutil.rmtree(os.path.join('./', dir))

    time.sleep(1)

    chunk_size = 100000
    num_chunks = (data.shape[0] + chunk_size - 1) // chunk_size
    # Save chunks to disk.
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(data))
        utils.save_df(data.iloc[start:end], f'data/Pogorelyy_YF/chunks/chunk_{i}.pkl')

    start_time = time.time()
    print(
        f'Perform clustering with a faiss_cluster_size of {faiss_cluster_size} and a training_sample_size of {training_sample_size} with a total of {total_sequences} sequences.')
    # Create clustering object.
    clustering = Clustering(faiss_training_data=meta['CDR3.amino.acid.sequence'], fitting_data_size=total_sequences,
                            max_sequence_size=max_seq_len, n_cpus=n_cpus, method='two-step',
                            faiss_cluster_size=faiss_cluster_size, mcl_params=[1.2, 2])

    print("Batch preclustering...")
    filenames = sorted(file for file in os.listdir('data/Pogorelyy_YF/chunks') if file.endswith('.pkl'))
    for file in filenames:
        f = pd.read_pickle(os.path.join('data/Pogorelyy_YF/chunks', file))
        clustering.batch_precluster(f['CDR3.amino.acid.sequence'])

    print("Batch clustering...")
    clusters = list(clustering.batch_cluster())

    print(f"Elapsed time: {time.time() - start_time} seconds")
    return clusters


def cluster_single(data: pd.Series, n_cpus: str = '8') -> ClusteringResult:
    """
    Performs clustering on a single sample of CDR3 amino acid sequences.

    :param n_cpus: The number of CPUs to use for clustering.
    :param data: A pandas DataFrame containing the CDR3 amino acid sequences to cluster.
    :return: A Clustering object containing the results of the clustering.
    """
    clustering = Clustering(n_cpus=n_cpus, method='two-step', mcl_params=[1.2, 2])
    return clustering.fit(data)


def individual(filename: str):
    data = pd.read_pickle(filename)

    # Preprocess the data.
    og_length = data.shape[0]
    # Remove sequences containing unresolved amino acids.
    # TODO - remove invalid char or whole sequence?
    data = data[~data['CDR3.amino.acid.sequence'].apply(contains_invalid_chars)]
    print(
        f"Removed {og_length - data.shape[0]} sequences containing invalid characters."
    )
    # # TODO - remove invalid peptide or whole sequence?
    # data = data[~data['CDR3.nucleotide.sequence'].apply(contains_stop_codon)]
    # og_length = data.shape[0]
    # # Remove sequences containing stop codons.
    # print(f"Removed {og_length - data.shape[0]} sequences containing stop codons.")

    # Remove sequences with read count less than 2.
    df = data.loc[data['Read.count'] > 1][:1000]

    # Only keep the CDR3 amino acid sequences as a Series.
    df: pd.DataFrame = df[['Read.count', 'CDR3.amino.acid.sequence']]

    # Remove duplicates.
    # TODO - update Read.count
    df = df.drop_duplicates(subset=['CDR3.amino.acid.sequence'])
    total_read_counts = np.sum(df['Read.count'])

    # Cluster the data.
    result = cluster_single(df['CDR3.amino.acid.sequence'])
    # Cluster the data in chunks.
    # result_ = cluster_chunks(data[:200])

    # Print clustering results.
    # print(result.summary())
    # print(result.clusters_df)

    start = time.time()

    # 3 clusters, 2 of size 2 and 1 of size 3
    # And when we remove duplicates we keep the motif of each cluster.
    # So 300 - (1 + 1 + 2) = 296

    # Group the sequences by their cluster label
    grouped = result.clusters_df.groupby('cluster')
    og_length = len(df)

    # Store the result of result.summary() in a variable
    summary = result.summary()

    # # Precompute the motifs for each cluster label
    # cluster_motifs = {
    #     cluster_label: f"cluster_{summary.loc[cluster_label]['motif']}"
    #     for cluster_label, _ in grouped
    # }
    #
    # cluster_read_counts = {
    #     cluster_label: sum(df.loc[df['CDR3.amino.acid.sequence'] == group.iloc[i]['junction_aa']]['Read.count'].iloc[0] for i in range(len(group)))
    #     for cluster_label, group in grouped
    # }

    cluster_data = {}
    for cluster_label, group in grouped:
        # compute cluster motif
        motif = f"cluster_{summary.loc[cluster_label]['motif']}"

        # compute cluster read count
        read_count = sum(
            df.loc[df['CDR3.amino.acid.sequence'] == group.iloc[i]['junction_aa']]['Read.count'].iloc[0] for i in
            range(len(group)))

        # add data to cluster_data dictionary
        cluster_data[cluster_label] = {'motif': motif, 'read_count': read_count}

    # Replace sequences in clusters with their motifs
    motif_dict = {
        seq: (cluster_data[cluster_label]['motif'], cluster_data[cluster_label]['read_count'])
        for cluster_label, group in grouped
        for seq in group['junction_aa']
    }

    # Update Read.count of clusters.
    df['Read.count'] = df.apply(
        lambda x: motif_dict[x['CDR3.amino.acid.sequence']][1] if x['CDR3.amino.acid.sequence'] in motif_dict else
        x['Read.count'], axis=1)
    df['CDR3.amino.acid.sequence'] = df['CDR3.amino.acid.sequence'].map(
        lambda x: x in motif_dict and motif_dict[x][0] or x)
    df = df.drop_duplicates(subset=['CDR3.amino.acid.sequence']).reset_index(drop=True)

    after_length = df.shape[0]
    print(after_length)
    predicted_length = og_length - len(result.cluster_contents())
    print(predicted_length)
    print(og_length)

    # Calculate term frequency (TF).
    # Divide the read count of each sequence by the total read count of all sequences in the dataset.
    # For a cluster, we already calculated the sum of the read counts of all sequences in the cluster.
    total_read_counts = np.sum(df['Read.count'])
    df['TF'] = df['Read.count'].map(lambda x: x / total_read_counts)

    # Calculate the inverse document frequency (IDF).
    # log(N + 1 / Df + 1) where N is the number of documents and n is the number of documents containing the term.

    # Calculate TF-IDF, for each sequence we multiply its TF by its IDF.

    print(f"Elapsed time: {time.time() - start} seconds")
    df.to_csv('clustered_pool4.csv', index=False)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
        individual('data/Pogorelyy_YF/P1_d0.pkl')
