import os
import multiprocessing as mp
import pickle
import re
import random
import time
import shutil
import warnings
import json

import joblib
import pandas as pd
import numpy as np
import requests
from collections import defaultdict
from scipy.sparse import vstack

from clustcr.clustering.clustering import ClusteringResult
from Bio import BiopythonDeprecationWarning
from typing import Tuple, List, Dict
from clustcr import Clustering
from functools import partial
from scipy.sparse import load_npz
from scipy.sparse import save_npz
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from sklearn.preprocessing import LabelEncoder

from gensim.corpora import MmCorpus
from tqdm import tqdm
from pandas.api.types import CategoricalDtype
from gensim import corpora, models

import utils
import tpm


def preprocess_files_sep(datadir: str, savedir: str, min_read_count: int, samples: int):
    file_list = [filename for filename in os.listdir(datadir) if
                 filename.endswith('.tsv') and filename.startswith('P')]
    # Only preprocess the samples that are in the hla_clusters.tsv file.
    samples_ = pd.read_csv(
        'data/emerson_preprocessed/hla_clusters.tsv', sep='\t'
    )['sample_name'].values
    file_list = [filename for filename in file_list if filename.split('_')[0] in samples_]
    file_list.sort()

    preprocess_files_parallel(file_list, datadir=datadir, savedir=savedir, samples=samples,
                              min_read_count=min_read_count)


@utils.timer_decorator
def preprocess_files_parallel(files: List[str], datadir: str, savedir: str, samples: int = 10,
                              min_read_count: int = 20):
    """
    Preprocess the files in parallel.

    :param files: List of files to preprocess.
    :param min_read_count: Minimum read count to filter on.
    :param samples: Amount of samples to preprocess.
    """
    files = files[:samples]
    num_processes = mp.cpu_count()  # Number of CPU cores

    with mp.Pool(processes=num_processes) as pool, \
            tqdm(total=len(files), desc='Preprocessing files') as pbar:
        arguments = [(filename, min_read_count, datadir, savedir) for filename in files]
        for _ in pool.imap_unordered(preprocess_file_parallel, arguments):
            pbar.update(1)


def preprocess_file_parallel(args):
    """
    Preprocess a single file in parallel.

    :param filename: Name of the file to preprocess.
    :param min_read_count: Minimum read count to filter on.
    """
    filename, min_read_count, datadir, savedir = args

    # skip files that have already been processed
    if os.path.exists(f'{savedir}/{filename.split(".")[0]}?minReadCount={min_read_count}.tsv'):
        return

    df = pd.read_csv(os.path.join(datadir, filename), sep='\t', index_col=0)
    preprocess_file(df, f'{savedir}/{filename.split(".")[0]}?minReadCount={min_read_count}.tsv',
                    sample_id=str(int(filename.split('_')[0].replace('P', ''))), min_read_count=min_read_count,
                    amino_acid_col='cdr3_amino_acid',
                    count_col='seq_reads', v_gene_col='v_resolved', j_gene_col='j_resolved',
                    nucleotide_col='cdr3_rearrangement')


@utils.timer_decorator
def preprocess_file(data, filename_save: str, sample_id: str = '', min_read_count: int = 2,
                    count_col: str = 'Read.count', amino_acid_col: str = 'CDR3.amino.acid.sequence',
                    nucleotide_col: str = 'CDR3.nucleotide.sequence',
                    v_gene_col: str = 'bestVGene', j_gene_col: str = 'bestJGene') -> None:
    """
    Preprocesses a file containing CDR3 amino acid sequences.

    :param sample_id:
    :param amino_acid_col:
    :param count_col:
    :param nucleotide_col:
    :param v_gene_col:
    :param j_gene_col:
    :param data: data to preprocess.
    :param filename_save: File to save the preprocessed data to.
    :param min_read_count: Minimum read count of a sequence to be included in the preprocessed data.
    """

    # Print out duplicates
    # duplicate_mask = data['CDR3.amino.acid.sequence'].duplicated(keep=False)
    # dupes = data[duplicate_mask].sort_values(by='CDR3.amino.acid.sequence')
    # print(dupes)
    # value_counts = data['CDR3.amino.acid.sequence'].value_counts()
    # print(value_counts[value_counts > 1])

    print(len(data))

    # Only keep the valuable columns.
    df: pd.DataFrame = (
        data[
            [
                count_col,
                amino_acid_col,
            ]
        ]
        if sample_id != ''
        else data[[count_col, amino_acid_col, 'sample_id']]
    )

    # Free memory.
    del data

    # # ===== Remove duplicates =====
    # og_length = df.shape[0]
    # # Sort by CDR3.amino.acid.sequence to group duplicates together.
    # # df = df.sort_values(['CDR3.amino.acid.sequence', 'bestVGene', 'bestJGene'])
    # df = df.sort_values([amino_acid_col])
    # # Remove duplicates and update Read.count.
    # # df = df.groupby(['CDR3.amino.acid.sequence', 'bestVGene', 'bestJGene'], as_index=False).agg({'Read.count': 'sum'})
    # df = df.groupby([amino_acid_col], as_index=False).agg({count_col: 'sum'})
    # print(f'Removed {og_length - df.shape[0]} duplicate sequences.')

    # ===== Remove duplicates within each sample =====
    og_length = df.shape[0]
    df = (
        df.sort_values(['cdr3_amino_acid'])
        if sample_id
        else df.sort_values(['sample_id', 'cdr3_amino_acid'])
    )
    df = (
        df.groupby(['cdr3_amino_acid'], as_index=False).agg(
            {'seq_reads': 'sum'}
        )
        if sample_id
        else df.groupby(['sample_id', 'cdr3_amino_acid'], as_index=False).agg(
            {'seq_reads': 'sum'}
        )
    )
    print(f'Removed {og_length - df.shape[0]} duplicate sequences.')

    # ===== Remove unresolved sequences =====
    og_length = df.shape[0]
    # Remove sequences that contain a stop codon, '*' in the sequence.
    df = df[~df[amino_acid_col].apply(lambda x: '*' in x)]
    print(
        f'Removed {og_length - df.shape[0]} sequences containing stop codon.'
    )

    # ===== Filter sequences on Read.count =====
    og_length = df.shape[0]
    df = df[df[count_col] >= min_read_count]
    print(f'Removed {og_length - df.shape[0]} sequences with Read.count < {min_read_count}.')

    if 'sample_id' not in df.columns:
        # Add sample_id column
        df['sample_id'] = sample_id

    print("Number of sequences after filtering: ", df.shape[0])
    # if df.shape[0] > 1000000:
    #     num_chunks = math.ceil(df.shape[0] / 150000)
    #     chunks = [df.iloc[i * 150000:(i + 1) * 150000] for i in range(num_chunks)]
    # else:
    #     chunks = [df]
    #
    # for i, chunk in enumerate(chunks):
    df.to_csv(f"{filename_save}", index=False, sep='\t')


@utils.timer_decorator
def term_doc_matrix(files: List[str], filename_save: str = '', chunksize: int = 10000):
    """
    Create a term-document matrix from the given files.

    :param filename_save: Filename to save the matrix to.
    :param files: List of files to create the matrix from.
    :param chunksize: The number of rows to read at a time.
    """
    if os.path.exists(filename_save):
        return

    # Define an empty DataFrame to store the concatenated data.
    concatenated_df = pd.DataFrame()

    # Iterate over all the files.
    for filename in tqdm(files, desc='Reading files'):
        # Read the file in chunks and concatenate each chunk to the main DataFrame.
        chunk_iter = pd.read_csv(filename, sep='\t', chunksize=chunksize)
        for chunk in chunk_iter:
            concatenated_df = pd.concat([concatenated_df, chunk])

    print("Number of sequences: ", concatenated_df.shape[0])

    # Create a categorical data type for cdr3_amino_acid column.
    cdr3_amino_acid_dtype = CategoricalDtype(categories=concatenated_df['cdr3_amino_acid'].unique(), ordered=True)

    # Use groupby and unstack instead of pivot_table
    dt_matrix = concatenated_df.groupby(['cdr3_amino_acid', 'sample_id'])['seq_reads'].sum().unstack(
        fill_value=0).reindex(cdr3_amino_acid_dtype.categories)

    print("Number of unique sequences: ", dt_matrix.shape[0])

    if filename_save != '':
        if filename_save.endswith('.pkl'):
            dt_matrix.to_pickle(filename_save)
        elif filename_save.endswith('.tsv'):
            dt_matrix.to_csv(filename_save, sep='\t')


@utils.timer_decorator
def efficient_doc_term_matrix(files: List[str], matrix_filename_save: str = '', sequences_filename_save: str = '',
                              chunksize: int = 10000):
    """
    Create a document-term matrix from the given files. This function is optimized for large files and uses sparse
    matrices for memory efficiency.

    :param matrix_filename_save: Filename to save the matrix to.
    :param sequences_filename_save: Filename to save the unique sequences to.
    :param files: List of files to create the matrix from.
    :param chunksize: The number of rows to read at a time.
    """
    if os.path.exists(matrix_filename_save) and os.path.exists(sequences_filename_save):
        return

    # Create LabelEncoders for sequences and documents
    seq_encoder = LabelEncoder()
    doc_encoder = LabelEncoder()

    # Define an empty DataFrame to store the concatenated data.
    all_data = pd.DataFrame()

    # Iterate over all the files.
    for filename in tqdm(files, desc='Reading files'):
        # Read the file in chunks and concatenate each chunk to the main DataFrame.
        chunk_iter = pd.read_csv(filename, sep='\t', chunksize=chunksize)
        for chunk in chunk_iter:
            all_data = pd.concat([all_data, chunk])

    # Remove duplicates and aggregate seq_reads across all data
    all_data_agg = all_data.groupby(['cdr3_amino_acid', 'sample_id'])['seq_reads'].sum().reset_index()

    # Transform sequences and documents to numerical IDs
    rows = seq_encoder.fit_transform(all_data_agg['cdr3_amino_acid'])
    cols = doc_encoder.fit_transform(all_data_agg['sample_id'])
    data = all_data_agg['seq_reads']

    # Create a sparse matrix
    dt_matrix = csr_matrix((data, (rows, cols)))

    # Save the matrix
    np.savez_compressed(matrix_filename_save, data=dt_matrix.data, indices=dt_matrix.indices,
                        indptr=dt_matrix.indptr, shape=dt_matrix.shape)

    # Save the unique sequences
    np.savez_compressed(sequences_filename_save, sequences=seq_encoder.classes_)

    # Save the encoders
    with open('seq_encoder.pkl', 'wb') as f:
        pickle.dump(seq_encoder, f)
    with open('doc_encoder.pkl', 'wb') as f:
        pickle.dump(doc_encoder, f)

    print("Number of unique sequences: ", len(seq_encoder.classes_))


def replace_sequences_with_cluster_labels(data_chunk, clusters_df):
    sequence_to_cluster = clusters_df.set_index('junction_aa')['cluster'].to_dict()
    # This function will run in a separate process and replace sequences with cluster labels in a chunk of data.
    data_chunk.index = pd.Series(data_chunk.index).replace(sequence_to_cluster).values

    return data_chunk


def process_cluster(args):
    # Retrieve arguments.
    document_term_matrix, clustering_result, file_index, tsv = args
    clusters_df = clustering_result.clusters_df
    summary = clustering_result.summary()
    start = time.time()

    print(clusters_df)
    print(summary)

    # Create mapping from sequence to cluster.
    sequence_to_cluster = clusters_df.set_index('junction_aa')['cluster'].to_dict()

    # Create a new series where the index is the unique values of 'clusters_df['cluster']' and the values are 'summary['motif']'.
    mapping_series = pd.Series(summary['motif'].values, index=clusters_df['cluster'].unique())

    # Map the 'cluster' in 'clusters_df' to 'motif' using the created mapping series.
    clusters_df['cluster_motif'] = clusters_df['cluster'].map(mapping_series)
    cluster_assignments = clusters_df[['junction_aa', 'cluster', 'cluster_motif']]
    cluster_assignments.columns = ['cdr3_amino_acid', 'cluster_index', 'cluster_motif']
    cluster_assignments.set_index('cluster_index', inplace=True)
    cluster_assignments.index = cluster_assignments.index.astype(str)

    print("Time to calculate clustering assignments: ", time.time() - start)

    return sequence_to_cluster, cluster_assignments


def process_cluster_(clustering_result):
    clusters_df = clustering_result.clusters_df
    start = time.time()

    print(clusters_df)

    # Create mapping from sequence to cluster.
    sequence_to_cluster = clusters_df.set_index('junction_aa')['cluster'].to_dict()

    print("Time to calculate clustering assignments: ", time.time() - start)

    return sequence_to_cluster


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

        # Parameters.
        DATADIR = 'data/emerson'
        SAVEDIR = 'data/emerson_preprocessed'
        # Parameters for choosing samples and filtering.
        SAMPLES: int = 668
        MIN_READ_COUNT: int = 75
        # Parameters for clustering.
        CLUSTERING_CHUNKS: bool = True
        CLUSTERING_CHUNK_SIZE: int = 100000
        CLUSTERING_CPU: int = 7
        CLUSTERING_USE_TSV: bool = False
        # Parameters for tpm.

        # ===== Preprocess the files separately =====
        # preprocess_files_sep(datadir=DATADIR, savedir=SAVEDIR, min_read_count=MIN_READ_COUNT, samples=SAMPLES)

        # ===== Create document-term matrix =====
        # Read in preprocessed files.
        # file_list = [os.path.join('data/emerson_preprocessed', filename) for filename in
        #              os.listdir('data/emerson_preprocessed') if filename.endswith('.tsv') and filename.startswith('P')]
        # file_list.sort()
        # file_list = file_list[:SAMPLES]
        # # Create document-term matrix, save as .pkl since there are too many rows.
        # term_doc_matrix(file_list, f'data/emerson_preprocessed/P0-P{SAMPLES}term_doc_matrix.pkl')
        # efficient_doc_term_matrix(file_list,
        #                           matrix_filename_save=f'data/emerson_preprocessed/P0-P{SAMPLES}doc_term_matrix.npz',
        #                           sequences_filename_save=f'data/emerson_preprocessed/P0-P{SAMPLES}sequences.npz')

        # first = pd.read_pickle('doc_term_matrix_.pkl')
        # # first.sort_values(inplace=True, by=1, ascending=False)
        # first.index = first.index.astype(str)
        # first.sort_index(inplace=True)
        # print(len(first))
        #
        # second = pd.read_csv('second.csv', index_col=0)
        # print(len(second))

        # print(first)
        # # first_ass = pd.read_pickle('cluster_assignments_.pkl')
        # # print(first_ass)
        # #
        # # second_ass = pd.read_pickle('all_cluster_assignments.pkl')
        # # print(second_ass)
        #
        # with np.load('new_dt_matrix.npz', allow_pickle=True) as loader:
        #     dt_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        #
        # Load the encoders

        # with open('doc_encoder.pkl', 'rb') as f:
        #     doc_encoder = pickle.load(f)
        #
        # # ===== Cluster sequences of document-term matrix =====
        # document_term_matrix = pd.read_pickle(f'data/emerson_preprocessed/P0-P{SAMPLES}term_doc_matrix.pkl')
        # print(document_term_matrix.rename_axis(
        #     'cdr3_amino_acid'
        # ).reset_index()[['cdr3_amino_acid']])
        #
        # sequences = np.load('P0-P4sequences.npz', allow_pickle=True)['sequences']
        #
        # a = pd.DataFrame(data=sequences,
        #                  columns=['cdr3_amino_acid'])
        #
        # print(a)
        #
        # # Clustering with chunks.
        # if CLUSTERING_CHUNKS:
        #     datadir = 'data/chunks/'
        #
        #     # Write the document-term matrix to chunks of specific size.
        #     # utils.write_dataframe_to_chunks(
        #     #     document_term_matrix.rename_axis(
        #     #         'cdr3_amino_acid'
        #     #     ).reset_index()[['cdr3_amino_acid']],
        #     #     CLUSTERING_CHUNK_SIZE,
        #     #     f'{datadir}chunk',
        #     #     datadir,
        #     # )
        #     utils.write_dataframe_to_chunks(
        #         document_term_matrix.rename_axis(
        #             'cdr3_amino_acid'
        #         ).reset_index()[['cdr3_amino_acid']],
        #         CLUSTERING_CHUNK_SIZE,
        #         f'{datadir}chunk',
        #         datadir,
        #     )
        #     # Calculate the total number of sequences in the dataset and save the chunk sized dataframes.
        #     sorted_files = sorted(os.listdir(datadir), key=lambda x: int(x.split('_')[1].split('.')[0].split('k')[1]))
        #     files = []
        #     total_cdr3s = 0
        #     for file in sorted_files:
        #         df = pd.read_csv(datadir + file, sep='\t')
        #         files.append(df)
        #         total_cdr3s += len(df)
        #     print(f"Total number of sequences: {total_cdr3s}.")
        #
        #     # Calculate the number of sequences to use for training.
        #     training_sample_size = round(1000 * (total_cdr3s / 5000))
        #     print(f"Training sample size: {training_sample_size}.")
        #
        #     # Create metareportoire.
        #     training_sample = utils.metareportoire(files, training_sample_size)
        #
        #     # Calculate the maximum sequence length.
        #     max_seq_len = training_sample[0]['cdr3_amino_acid'].str.len().max()
        #     print(f"Max sequence length: {max_seq_len}.")
        #
        #     # Initialize the clustering object.
        #     clustering = Clustering(faiss_training_data=training_sample[0]['cdr3_amino_acid'],
        #                             fitting_data_size=total_cdr3s,
        #                             max_sequence_size=max_seq_len,
        #                             n_cpus=CLUSTERING_CPU)
        #
        #     print("Clustering initialized.")
        #
        #     # Perform pre-clustering.
        #     for i in range(len(files)):
        #         print(f"Pre-clustering file: {str(i)}.")
        #         clustering.batch_precluster(files[i]['cdr3_amino_acid'])
        #
        #     # Load the term-document matrix, sequences, and encoders
        #     with np.load('P0-P4doc_term_matrix.npz', allow_pickle=True) as loader:
        #         dt_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        #
        #     encoded_sequences = np.load('P0-P4sequences.npz', allow_pickle=True)['sequences']
        #     with open('seq_encoder.pkl', 'rb') as f:
        #         seq_encoder = pickle.load(f)
        #     with open('doc_encoder.pkl', 'rb') as f:
        #         doc_encoder = pickle.load(f)
        #
        #     # Retrieve the clustering results.
        #     with mp.Pool(processes=mp.cpu_count()) as pool:
        #         results = pool.map(process_cluster_, clustering.batch_cluster())
        #
        #     # Combine results from multiprocessing.
        #     sequence_to_cluster = {}
        #     for res in results:
        #         sequence_to_cluster |= res
        #
        #     start = time.time()
        #
        #     # Find the max encoded term to use as a base for new encoding for clusters
        #     max_encoded_term = len(seq_encoder.classes_)
        #
        #     # Create a mapping from encoded terms to their corresponding clusters
        #     encoded_term_to_cluster = {seq_encoder.transform([term])[0]: cluster for term, cluster in
        #                                sequence_to_cluster.items()}
        #
        #     # List of lists to store term-document-frequency tuples for each unique cluster
        #     clusters_data = defaultdict(list)
        #     for i, j, v in zip(dt_matrix.row, dt_matrix.col, dt_matrix.data):
        #         if i in encoded_term_to_cluster:
        #             cluster = encoded_term_to_cluster[i]
        #         else:
        #             cluster = i
        #         clusters_data[cluster].append((i, j, v))
        #
        #     # Convert each cluster's data into a sparse matrix and stack them vertically
        #     matrix_list = []
        #     for cluster, term_tuples in clusters_data.items():
        #         rows, cols, data = zip(*term_tuples)
        #         matrix = csr_matrix((data, (np.zeros_like(rows), cols)), shape=(1, dt_matrix.shape[1]))
        #         matrix_list.append(matrix)
        #
        #     new_dt_matrix = vstack(matrix_list)
        #
        #     pd.DataFrame(new_dt_matrix.toarray()).to_csv('second.csv')
        #
        #     print("Time taken: ", time.time() - start)

        #
        #     # # Retrieve the clustering results.
        #     # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     #     results = pool.map(process_cluster,
        #     #                        ((document_term_matrix.copy(), result, i, CLUSTERING_USE_TSV) for i, result in
        #     #                         enumerate(clustering.batch_cluster())))
        #     #
        #     # # Combine results from multiprocessing.
        #     # sequence_to_cluster = {}
        #     # for result in results:
        #     #     sequence_to_cluster |= result[0]
        #     # all_cluster_assignments = [result[1] for result in results]
        #     # all_cluster_assignments = pd.concat(all_cluster_assignments)
        #     #
        #     # # Remove duplicates inside document_term_matrix and aggregate the counts.
        #     # index = pd.Index(document_term_matrix.index)
        #     # document_term_matrix.index = index.to_series().map(lambda x: sequence_to_cluster.get(x, x))
        #     # all_document_cluster_matrix = document_term_matrix.groupby(document_term_matrix.index).sum()
        #     #
        #     # print(f'Found {int(all_cluster_assignments.index[-1])} clusters.')
        #     #
        #     # print(
        #     #     f'Replaced {len(sequence_to_cluster) - int(all_cluster_assignments.index[-1])} sequences with cluster labels.')
        #
        #     # Retrieve the clustering results.
        #     with mp.Pool(processes=mp.cpu_count()) as pool:
        #         results = pool.map(process_cluster__,
        #                            ((encoded_sequences, result, i) for i, result in
        #                             enumerate(clustering.batch_cluster())))
        #
        #     # Combine results from multiprocessing.
        #     encoded_sequences = np.hstack(results)
        #     print(encoded_sequences)
        #
        # # Write to disk.
        # if CLUSTERING_USE_TSV:
        #     all_document_cluster_matrix.to_csv('data/emerson_preprocessed/doc_term_matrix.tsv', index=True,
        #                                        sep='\t')
        #     all_cluster_assignments.to_csv('data/emerson_preprocessed/cluster_assignments.tsv', index=True,
        #                                    sep='\t')
        # else:
        #     all_cluster_assignments.to_pickle('data/emerson_preprocessed/cluster_assignments.pkl')
        #     all_document_cluster_matrix.to_pickle('data/emerson_preprocessed/doc_term_matrix.pkl')
        #
        # # Clean up.
        # clustering.batch_cleanup()
        # # Remove the chunked dataframes
        # for file in os.listdir('data/emerson_preprocessed/'):
        #     if file.startswith('cluster_assignments_') or file.startswith('doc_term_matrix_'):
        #         os.remove(os.path.join('data/emerson_preprocessed/', file))

        # else:
        #     result = cluster_single(document_term_matrix.rename_axis('cdr3_amino_acid').reset_index(),
        #                             cdr3_col='cdr3_amino_acid', n_cpus=CLUSTERING_CPU)
        #
        #     sequence_to_cluster, cluster_assignments = process_cluster(
        #         (document_term_matrix, result, None, CLUSTERING_USE_TSV))
        #
        #     # Remove duplicates inside document_term_matrix and aggregate the counts.
        #     index = pd.Index(document_term_matrix.index)
        #     document_term_matrix.index = index.to_series().map(lambda x: sequence_to_cluster.get(x, x))
        #     document_term_matrix = document_term_matrix.groupby(document_term_matrix.index).sum()
        #
        #     # Write to disk.
        #     if CLUSTERING_USE_TSV:
        #         document_term_matrix.to_csv('data/emerson_preprocessed/doc_term_matrix.tsv', index=True,
        #                                     sep='\t')
        #         cluster_assignments.to_csv('data/emerson_preprocessed/cluster_assignments.tsv', index=True,
        #                                    sep='\t')
        #     else:
        #         cluster_assignments.to_pickle('data/emerson_preprocessed/cluster_assignments.pkl')
        #         document_term_matrix.to_pickle('data/emerson_preprocessed/doc_term_matrix.pkl')
        #
        #     print(
        #         f'Found {len(result.summary())} clusters with an average size of {round(result.summary()["size"].mean(), 2)}.')

        # ===== Topic Modelling =====
        tdm = pd.read_pickle('data/emerson_preprocessed/P0-P666doc_term_matrix_clustered?minReadCount=200.pkl')
        samples_ = pd.read_csv(
            'data/emerson_preprocessed/hla_clusters.tsv', sep='\t'
        )['sample_name'].values
        samples_ = [int(i.split("P")[1]) for i in samples_]
        tdm.drop(columns=[col for col in tdm if col not in samples_], inplace=True)

        # Transpose the matrix so that rows are documents (samples) and columns are terms (sequences).
        tdm = tdm.T

        print(tdm)

        # Create Dictionary and Corpus.
        # tpm.create_dictionary(tdm, 'data/emerson_preprocessed/P0-P666doc_term_matrix_clustered?minReadCount=200.dict')
        dictionary: corpora.Dictionary = corpora.Dictionary.load(
            'data/emerson_preprocessed/P0-P666doc_term_matrix_clustered?minReadCount=200.dict')
        # tpm.create_corpus(tdm, dictionary, 'data/emerson_preprocessed/P0-P666doc_term_matrix_clustered?minReadCount=200.mm')
        corpus: corpora.MmCorpus = corpora.MmCorpus(
            'data/emerson_preprocessed/P0-P666doc_term_matrix_clustered?minReadCount=200.mm')

        print(f'{corpus.num_docs} samples/documents inside the corpus.')
        print(f'{corpus.num_terms} sequences/terms inside the corpus.')

        # Train LDA model.
        # model = tpm.train_model(corpus, dictionary, 'data/emerson_preprocessed/P0-P666doc_term_matrix_clustered?minReadCount=200.model', num_topics=15, chunksize=2000, iterations=1500, passes=100)

        # # Find the optimal parameters (duurt veel te lang)
        # # tpm.optimize_parameters(tdm, corpus, dictionary, max_topics=25)
        #
        # Load model.
        model: models.ldamodel.LdaModel = models.ldamodel.LdaModel.load(
            'data/emerson_preprocessed/P0-P666doc_term_matrix_clustered?minReadCount=200.model')

        # Get the most dominant topics for each sample/document.
        doc_topics = [model.get_document_topics(doc) for doc in corpus]
        # Convert to dataframe.
        doc_topics_df = pd.DataFrame(doc_topics)
        # Write to disk.
        # doc_topics_df.to_csv('data/emerson_preprocessed/P0-P100doc_topics.tsv', sep='\t', index=False)
        # doc_topics_df.to_csv('data/emerson_preprocessed/P0-P600doc_topics.tsv', sep='\t', index=False)

        # Get all topics and their probabilities for each sample/document.
        topics_list = [[(topic_id, round(prob, 3)) for topic_id, prob in model.get_document_topics(bow)] for bow in
                       corpus]
        topic_df = pd.DataFrame(topics_list, index=tdm.index)
        topic_df.index = topic_df.index.astype(int)

        # Load cluster assignments.
        cluster_assignments = pd.read_pickle('data/emerson_preprocessed/P0-P666cluster_assignments?minReadCount=200.pkl')
        # cluster_assignments = pd.read_pickle(
        #     'data/emerson_preprocessed/P0-P100cluster_assignments.pkl')

        # Print each topic.
        for topic_id in range(model.num_topics):
            print(f"Topic #{topic_id}")
            # Get most probable sequences for this topic.
            for sequence, prob in model.show_topic(topic_id):

                # If sequence is a cluster label, print the cluster contents.
                if sequence in cluster_assignments.index:
                    rows = cluster_assignments.loc[cluster_assignments.index == str(sequence)]
                    print(f"\tCluster: {rows['cluster_motif'][0]}, Probability: {prob:.4f}")
                    print(f"\t\tCluster Contents: {', '.join(rows['cdr3_amino_acid'])}")
                else:
                    print(f"\tSequence: {sequence}, Probability: {prob:.4f}")

        # Read metadata from disk.
        # metadata = pd.read_csv('data/emerson/metadata_merged.tsv', sep='\t')
        # metadata['sample_id'] = metadata['sample_id'].astype(int)

        # Join this DataFrame with your metadata DataFrame.
        # metadata_with_topics = metadata.merge(topic_df, left_on='sample_id', right_index=True).drop(
        #     columns=['sample_id']).reindex(
        #     columns=['sample_name', 'species', 'Age', 'Biological Sex', 'Ethnic Group', 'Racial Group',
        #              'Virus Diseases', 'locus', 'product_subtype'] + list(topic_df.columns) + ['hla_class_i',
        #                                                                                        'hla_class_ii',
        #                                                                                        'sample_amount_ng',
        #                                                                                        'sample_cells_mass_estimate',
        #                                                                                        'counting_method',
        #                                                                                        'primer_set']).sort_values(
        #     by='sample_name')
        # print(metadata_with_topics)
        # metadata_with_topics.to_csv('.tsv', sep='\t', index=False)

        hla_clusters = pd.read_csv('data/emerson_preprocessed/hla_clusters.tsv', sep='\t')
        hla_clusters['sample_name'] = hla_clusters['sample_name'].str.split('P').str[1].astype(int)

        merged_df = hla_clusters.merge(topic_df, left_on='sample_name', right_index=True)

        merged_df['topic'] = merged_df[0].apply(lambda x: x[0])

        umap_cluster_counts = merged_df.groupby('umap_hdb_cluster_labels')['topic'].value_counts()

        umap_cluster_percentages = umap_cluster_counts.groupby(level=0).apply(lambda x: x / x.sum() * 100)

        umap_cluster_percentages.to_csv('data/emerson_preprocessed/results.tsv', sep='\t')

        print(umap_cluster_percentages)


        # # Load the term-document matrix, sequences, and encoders
        # with np.load('data/emerson_preprocessed/P0-P668doc_term_matrix.npz', allow_pickle=True) as loader:
        #     td_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        #
        # # encoded_sequences = np.load('P0-P4sequences.npz', allow_pickle=True)['sequences']
        # with open('seq_encoder.pkl', 'rb') as f:
        #     seq_encoder = pickle.load(f)
        # with open('doc_encoder.pkl', 'rb') as f:
        #     doc_encoder = pickle.load(f)
        #
        # dtm = td_matrix.transpose()
        #
        # # Create Dictionary and Corpus.
        # # tpm.create_dictionary_(seq_encoder, 'dictionary600.dict')
        # dictionary: corpora.Dictionary = corpora.Dictionary.load('dictionary600.dict')
        # print(len(dictionary))
        # # tpm.create_corpus_(dtm, dictionary, filename='corpuss600_.mm')
        # corpus: corpora.MmCorpus = corpora.MmCorpus('corpuss600.mm')
        #
        # print(f'{corpus.num_docs} samples/documents inside the corpus.')
        # print(f'{corpus.num_terms} sequences/terms inside the corpus.')
        #
        # # Train LDA model.
        # model = tpm.train_model(corpus, dictionary, 'model600_4topics.model', num_topics=4, chunksize=5000, iterations=1000,
        #                         passes=50)
        #
        # Find the optimal parameters (duurt veel te lang)
        # tpm.optimize_parameters(tdm, corpus, dictionary, max_topics=25)

        # Load model.
        # model: models.ldamodel.LdaModel = models.ldamodel.LdaModel.load('model600_4topics.model')
        #
        # # Get the most dominant topics for each sample/document.
        # doc_topics = [model.get_document_topics(doc) for doc in corpus]
        # # Convert to dataframe.
        # doc_topics_df = pd.DataFrame(doc_topics)
        # # Write to disk.
        # doc_topics_df.to_csv('topics600_4topics.tsv', sep='\t', index=False)
        # # doc_topics_df.to_csv('data/emerson_preprocessed/P0-P600doc_topics.tsv', sep='\t', index=False)
        #
        # # Get all topics and their probabilities for each sample/document.
        # topics_list = [[(topic_id, round(prob, 3)) for topic_id, prob in model.get_document_topics(bow)] for bow in
        #                corpus]
        # topic_df = pd.DataFrame(topics_list, index=doc_encoder.classes_)
        # topic_df.index = topic_df.index.astype(int)
        #
        # # Print each topic.
        # for topic_id in range(model.num_topics):
        #     print(f"Topic #{topic_id}")
        #     # Get most probable sequences for this topic.
        #     for sequence, prob in model.show_topic(topic_id):
        #         print(f"\tSequence: {sequence}, Probability: {prob:.4f}")
        #
        # # Read metadata from disk.
        # metadata = pd.read_csv('data/emerson/metadata_merged.tsv', sep='\t')
        # metadata['sample_id'] = metadata['sample_id'].astype(int)
        #
        # # Join this DataFrame with your metadata DataFrame.
        # metadata_with_topics = metadata.merge(topic_df, left_on='sample_id', right_index=True).drop(
        #     columns=['sample_id']).reindex(
        #     columns=['sample_name', 'species', 'Age', 'Biological Sex', 'Ethnic Group', 'Racial Group',
        #              'Virus Diseases', 'locus', 'product_subtype'] + list(topic_df.columns) + ['hla_class_i',
        #                                                                                        'hla_class_ii',
        #                                                                                        'sample_amount_ng',
        #                                                                                        'sample_cells_mass_estimate',
        #                                                                                        'counting_method',
        #                                                                                        'primer_set']).sort_values(
        #     by='sample_name')
        #
        # metadata_with_topics.to_csv('metadata_with_topics_Vee_4topics.tsv', sep='\t', index=False)
        #
        # df = pd.read_csv('metadata_with_topics_Veel.tsv', sep='\t')
        # # Drop nan values
        # df = df.dropna(subset=['Biological Sex']).reset_index(drop=True)
        # # df = df[df['Ethnic Group'] != 'Unknown Ethnicity']
        # # df.reset_index(drop=True, inplace=True)
        #
        # # Rename column to topic.
        # df = df.rename(columns={'0': 'topic'})
        #
        # topic_assignments = df['topic']
        #
        # extract_topic = lambda assignment: assignment.split(',')[0].strip('() ')
        #
        # topic_assignments = df['topic']
        # biological_sex = df['Biological Sex']
        #
        # topic_mapping = {}
        #
        # for i in range(len(df)):
        #     assigned_topic = extract_topic(topic_assignments[i])
        #     sex = biological_sex[i]
        #
        #     if assigned_topic in topic_mapping:
        #         topic_mapping[assigned_topic][sex] = topic_mapping[assigned_topic].get(sex, 0) + 1
        #     else:
        #         topic_mapping[assigned_topic] = {sex: 1}
        #
        # print(topic_mapping)
        #
        # # Determine the dominant topic interpretation for each sex
        # dominant_topics = {}
        # for sex in set(biological_sex):
        #     dominant_topic = max(topic_mapping, key=lambda topic: topic_mapping[topic].get(sex, 0))
        #     dominant_topics[sex] = dominant_topic
        #
        # print(dominant_topics)
        #
        # # Calculate the accuracy of topic assignments for each sex
        # accurate_assignments = 0
        # for i in range(len(df)):
        #     assigned_topic = extract_topic(topic_assignments[i])
        #     sex = biological_sex[i]
        #
        #     if assigned_topic == dominant_topics[sex]:
        #         accurate_assignments += 1
        #
        # accuracy = accurate_assignments / len(df) * 100
        #
        # print(f"Accuracy: {accuracy:.2f}%")
        #
        # print(df.columns)
        # print(df)
