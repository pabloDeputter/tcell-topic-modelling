import os
import multiprocessing as mp
import re
import random
import time
import shutil
import warnings
import json

import pandas as pd
import numpy as np
import requests

from clustcr.clustering.clustering import ClusteringResult
from Bio import BiopythonDeprecationWarning
from typing import Tuple, List
from clustcr import Clustering
from functools import partial
from tqdm import tqdm
from pandas.api.types import CategoricalDtype
from gensim import corpora, models

import utils


def contains_stop_codon(s: str) -> bool:
    """
    Check whether a given DNA sequence contains any stop codons.

    :param s: A string representing a DNA sequence.
    :return: A boolean value indicating whether the input DNA sequence contains a stop codon.
             Returns True if a stop codon is found, False otherwise.
    """
    stop_codons = ['TAA', 'TAG', 'TGA']
    return any(s[i:i + 3] in stop_codons for i in range(0, len(s) - 2, 3))


def check_stop_codon(s: pd.Series, print_results: bool = False) -> bool:
    """
    Check if the given pandas Series contains a stop codon in the 'CDR3.nucleotide.sequence' column.
    Indicate which stop codons and amino acids are affected in the 'CDR3.amino.acid.sequence' column.

    :param s: A pandas Series containing the required columns.
    :param print_results: A boolean indicating whether to print the results. Default is False.
    :return: A boolean indicating the presence of a stop codon if print_results is False.
    """
    nucleotide_sequence = s['CDR3.nucleotide.sequence']
    amino_acid_sequence = s['CDR3.amino.acid.sequence']

    # Define stop codons.
    stop_codons = ['TAA', 'TAG', 'TGA']
    stop_codon_positions = []
    affected_amino_acids = []

    # Traverse nucleotide sequence in steps of 3.
    for i in range(0, len(nucleotide_sequence) - 2, 3):
        codon = nucleotide_sequence[i:i + 3]
        if codon in stop_codons:
            # Store the stop codon and its position, we use // 3 to get the position of the affected amino acid.
            stop_codon_positions.append((codon, i // 3))

    if len(stop_codon_positions) > 0:
        for codon, position in stop_codon_positions:
            affected_amino_acids.append(amino_acid_sequence[position])

        if print_results:
            print("Stop codons found:")
            for codon, position in stop_codon_positions:
                print(codon, "at position", position)
            print("Amino acids affected:", affected_amino_acids)
        else:
            return any(aa == '*' for aa in affected_amino_acids)
    return False


def contains_invalid_chars(seq: str) -> bool:
    """
    Check if a given CDR3 amino acid sequence contains any invalid characters.

    :param seq: A string representing a CDR3 amino acid sequence.
    :return: A boolean value indicating whether the input sequence contains any invalid characters.
             Returns True if invalid characters are found, False otherwise.
    """
    valid_chars_regex = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY_]+$')
    return bool(valid_chars_regex.match(seq))


def write_dataframe_to_chunks(df, chunk_size, filename_prefix, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Remove everything in the output directory.
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    num_chunks = len(df) // chunk_size + 1

    for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk = df.iloc[chunk_start:chunk_end]

        filename = f"{filename_prefix}_chunk{i + 1}.tsv"
        chunk.to_csv(filename, index=False, sep='\t')
        print(f"Chunk {i + 1} saved to {filename}")


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


def cluster_chunks(data: pd.DataFrame, n_cpus: str = '8', cdr3_col='CDR3.amino.acid.sequence'):
    """
    Performs clustering on a large dataset of CDR3 amino acid sequences by dividing it into smaller
    chunks and clustering each chunk independently.

    :param cdr3_col:
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
    meta = data[[cdr3_col]].copy()
    # Remove duplicates.
    meta.drop_duplicates(inplace=True)
    # Randomly sample the metareportoire.
    if len(meta) > training_sample_size:
        meta = meta.sample(training_sample_size)
    # Compute the maximum sequence length in the metareportoire.
    max_seq_len = meta[cdr3_col].str.len().max()

    # Create a dictionary for chunks.
    if not os.path.exists('data//chunks'):
        os.makedirs('data//chunks')
    # Clear .pkl files in 'chunks' directory.
    [os.remove(os.path.join(dirpath, file)) for dirpath, dirnames, filenames in os.walk('data/chunks') for
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
        utils.save_df(data.iloc[start:end], f'data/chunks/chunk_{i}.pkl')

    start_time = time.time()
    print(
        f'Perform clustering with a faiss_cluster_size of {faiss_cluster_size} and a training_sample_size of {training_sample_size} with a total of {total_sequences} sequences.')
    # Create clustering object.
    clustering = Clustering(faiss_training_data=meta[cdr3_col], fitting_data_size=total_sequences,
                            max_sequence_size=max_seq_len, n_cpus=n_cpus, method='two-step',
                            faiss_cluster_size=faiss_cluster_size, mcl_params=[1.2, 2])

    print("Batch preclustering...")
    filenames = sorted(file for file in os.listdir('data/chunks') if file.endswith('.pkl'))
    for file in filenames:
        f = pd.read_pickle(os.path.join('data/chunks', file))
        clustering.batch_precluster(f[cdr3_col])

    print("Batch clustering...")
    clusters = list(clustering.batch_cluster())

    print(f"Elapsed time: {time.time() - start_time} seconds")
    return clusters


def cluster_single(data: pd.DataFrame, n_cpus='8',
                   cdr3_col='CDR3.amino.acid.sequence', v_gene_col='bestVGene',
                   include_vgene: bool = False) -> ClusteringResult:
    """
    Performs clustering on a single sample of CDR3 amino acid sequences.

    :param data: Data containing the CDR3 amino acid sequences to cluster.
    :param n_cpus: The number of CPUs to use for clustering.
    :param v_gene_col: Column name of the V-gene column.
    :param cdr3_col: Column name of the CDR3 amino acid sequence column.
    :param include_vgene: Include V-gene information in the clustering.
    :return: A Clustering object containing the results of the clustering.
    """
    clustering = Clustering(n_cpus=8, method='two-step', mcl_params=[1.2, 2])
    if include_vgene:
        return clustering.fit(data[[cdr3_col, v_gene_col]], include_vgene=True, cdr3_col=cdr3_col,
                              v_gene_col=v_gene_col)
    else:
        return clustering.fit(data[cdr3_col])


def cluster(df: pd.DataFrame, filename_save: str, count_col: str = 'Read.count',
            amino_acid_col: str = 'CDR3.amino.acid.sequence'):
    # result_ = cluster_single(df, include_vgene=True) # Doesn't work properly and way too slow
    result = cluster_single(df, cdr3_col=amino_acid_col)
    # utils.visualize_features(result.compute_features())
    clusters_df = result.clusters_df
    summary = result.summary()

    print(f'Found {len(summary)} clusters with an average size of {round(summary["size"].mean(), 2)}.')

    # ===== Cluster assignments =====
    start = time.time()
    # Group sequences by their cluster and compute total cluster Read.count.
    clusters_df[count_col] = clusters_df['junction_aa'].map(df.set_index(amino_acid_col)[count_col])
    cluster_data = clusters_df.groupby('cluster').agg({'junction_aa': 'first', count_col: 'sum'}).reset_index()

    # Calculate term frequency (TF) for each cluster.
    # Divide the read count of each sequence by the total read count of all sequences in the dataset.
    total_read_counts = df[count_col].sum()
    cluster_data['TF'] = ((cluster_data[count_col] / total_read_counts).astype(float) * 100)

    # Replace sequences with cluster motif.
    cluster_data[amino_acid_col] = cluster_data.apply(
        lambda row: f"{summary.loc[row['cluster']]['motif']}", axis=1)

    # Merge original sequences with cluster data.
    merged_data = df.set_index(amino_acid_col).join(cluster_data.set_index('junction_aa'),
                                                    rsuffix='_cluster')

    # Replace values with cluster data.
    merged_data[amino_acid_col] = np.where(merged_data[amino_acid_col].isnull(),
                                           merged_data.index,
                                           merged_data[amino_acid_col])
    merged_data[count_col] = np.where(merged_data[f'{count_col}_cluster'].isnull(), merged_data[count_col],
                                      merged_data[f'{count_col}_cluster'])

    # Calculate term frequency (TF) for each sequence.
    merged_data['TF'] = (merged_data[count_col] / total_read_counts).astype(float)
    df = merged_data[[count_col, amino_acid_col, 'TF']].reset_index(drop=True).sort_values(
        by='TF', ascending=False)

    print(f"Elapsed time: {time.time() - start} seconds")
    df.to_csv(filename_save, index=False, sep='\t')


@utils.timer_decorator
def preprocess_file(data, filename_save: str, sample_id: str = '', min_read_count: int = 2,
                    count_col: str = 'Read.count', amino_acid_col: str = 'CDR3.amino.acid.sequence',
                    nucleotide_col: str = 'CDR3.nucleotide.sequence',
                    v_gene_col: str = 'bestVGene', j_gene_col: str = 'bestJGene', do_cluster: bool = False) -> None:
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

    # ===== Cluster the sequences =====
    if do_cluster:
        cluster(df, filename_save, count_col, amino_acid_col)


def search_sequences(sequences: pd.Series):
    """
    Search for sequences in the VDJdb database.

    :param sequences: Sequences to search for.
    """
    for seq in sequences:
        print("Searching for sequence: ", seq)
        response = requests.post('https://vdjdb.cdr3.net/api/database/search', data=json.dumps({
            "filters": [
                {
                    "column": "cdr3",
                    "value": seq,
                    "filterType": "pattern",
                    "negative": False
                }
            ]
        }), headers={
            "Content-Type": "application/json"
        })
        print(response.json())
        if response.json()['recordsFound'] >= 1:
            print("Found sequence: ", seq)
        time.sleep(0.5)


@utils.timer_decorator
def preprocess_files_individually(files: List[str], samples: int = 10, min_read_count: int = 20):
    """
    Preprocess the files individually.

    :param files: List of files to preprocess.
    :param min_read_count: Minimum read count to filter on.
    :param samples: Amount of samples to preprocess.
    """
    files = files[:samples]
    for filename in tqdm(files, desc='Preprocessing files'):
        # skip files that have already been processed
        if os.path.exists(f'data/emerson_preprocessed/{filename.split(".")[0]}?minReadCount={min_read_count}.tsv'):
            continue

        df = pd.read_csv(os.path.join('data/emerson', filename), sep='\t', index_col=0)
        preprocess_file(df, f'data/emerson_preprocessed/{filename.split(".")[0]}?minReadCount={min_read_count}.tsv',
                        sample_id=str(int(filename.split('_')[0].replace('P', ''))), min_read_count=min_read_count,
                        amino_acid_col='cdr3_amino_acid',
                        count_col='seq_reads', v_gene_col='v_resolved', j_gene_col='j_resolved',
                        nucleotide_col='cdr3_rearrangement', do_cluster=False)


@utils.timer_decorator
def doc_term_matrix(files: List[str], filename_save: str = '', chunksize: int = 10000):
    """
    Create a document-term matrix from the given files.

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
def preprocess_files_parallel(files: List[str], samples: int = 10, min_read_count: int = 20):
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
        arguments = [(filename, min_read_count) for filename in files]
        for _ in pool.imap_unordered(preprocess_file_parallel, arguments):
            pbar.update(1)


def preprocess_file_parallel(args):
    """
    Preprocess a single file in parallel.

    :param filename: Name of the file to preprocess.
    :param min_read_count: Minimum read count to filter on.
    """
    filename, min_read_count = args

    # skip files that have already been processed
    if os.path.exists(f'data/emerson_preprocessed/{filename.split(".")[0]}?minReadCount={min_read_count}.tsv'):
        return

    df = pd.read_csv(os.path.join('data/emerson', filename), sep='\t', index_col=0)
    preprocess_file(df, f'data/emerson_preprocessed/{filename.split(".")[0]}?minReadCount={min_read_count}.tsv',
                    sample_id=str(int(filename.split('_')[0].replace('P', ''))), min_read_count=min_read_count,
                    amino_acid_col='cdr3_amino_acid',
                    count_col='seq_reads', v_gene_col='v_resolved', j_gene_col='j_resolved',
                    nucleotide_col='cdr3_rearrangement', do_cluster=False)


def replace_sequences_with_cluster_labels(data_chunk, clusters_df):
    sequence_to_cluster = clusters_df.set_index('junction_aa')['cluster'].to_dict()
    # This function will run in a separate process and replace sequences with cluster labels in a chunk of data.
    data_chunk.index = pd.Series(data_chunk.index).replace(sequence_to_cluster).values

    return data_chunk


def process_cluster(document_term_matrix: pd.DataFrame, clusters_df, summary, file_index: None, tsv=False):
    start = time.time()

    # If there are less than 1000 clusters, we can the clustering assignments without multiprocessing.
    if len(clusters_df) <= 1000:
        # Replace sequences with cluster labels.
        sequence_to_cluster = clusters_df.set_index('junction_aa')['cluster'].to_dict()
        document_term_matrix.index = pd.Series(document_term_matrix.index).replace(sequence_to_cluster).values

    else:
        # Split the document_term_matrix into chunks, each of which will be processed by a separate process.
        n_cpus = mp.cpu_count()
        chunk_size = len(document_term_matrix) // n_cpus
        data_chunks = [document_term_matrix[i:i + chunk_size] for i in
                       range(0, len(document_term_matrix), chunk_size)]

        with mp.Pool(n_cpus) as pool:
            replaced_data_chunks = pool.starmap(replace_sequences_with_cluster_labels,
                                                [(chunk, clusters_df) for chunk in data_chunks])

        # Combine the chunks back into a single DataFrame.
        document_term_matrix = pd.concat(replaced_data_chunks)

    # Group by cluster and sum the occurrences.
    document_cluster_matrix = document_term_matrix.groupby(document_term_matrix.index).sum()

    # Store the cluster assignments.
    clusters_df['cluster_motif'] = clusters_df['cluster'].map(summary['motif'])
    cluster_assignments = clusters_df[['junction_aa', 'cluster', 'cluster_motif']]
    cluster_assignments.columns = ['cdr3_amino_acid', 'cluster_index', 'cluster_motif']
    cluster_assignments.set_index('cluster_index', inplace=True)
    cluster_assignments.index = cluster_assignments.index.astype(str)
    document_cluster_matrix.index = document_cluster_matrix.index.astype(str)

    print("Time to calculate clustering assignments: ", time.time() - start)

    # Write to disk.
    if file_index is None:
        if tsv:
            document_cluster_matrix.to_csv('data/emerson_preprocessed/doc_term_matrix.tsv', sep='\t')
            cluster_assignments.to_csv('data/emerson_preprocessed/cluster_assignments.tsv', index=True, sep='\t')
        else:
            document_cluster_matrix.to_pickle('data/emerson_preprocessed/doc_term_matrix.pkl')
            cluster_assignments.to_pickle('data/emerson_preprocessed/cluster_assignments.pkl')
    elif tsv:
        document_cluster_matrix.to_csv(f'data/emerson_preprocessed/doc_term_matrix_{file_index}.tsv', sep='\t')
        cluster_assignments.to_csv(f'data/emerson_preprocessed/cluster_assignments_{file_index}.tsv', index=True,
                                   sep='\t')
    else:
        document_cluster_matrix.to_pickle(f'data/emerson_preprocessed/doc_term_matrix_{file_index}.pkl')
        cluster_assignments.to_pickle(f'data/emerson_preprocessed/cluster_assignments_{file_index}.pkl')


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

        # Parameters for choosing samples and filtering.
        SAMPLES: int = 100
        MIN_READ_COUNT: int = 50
        # Parameters for clustering.
        CLUSTERING_CHUNKS: bool = True
        CLUSTERING_CHUNK_SIZE: int = 100000
        CLUSTERING_CPU: int = 7
        CLUSTERING_USE_TSV: bool = False

        # # ===== Preprocess the files separately =====
        # file_list = [filename for filename in os.listdir('data/emerson') if
        #              filename.endswith('.tsv') and filename.startswith('P')]
        # file_list.sort()
        # preprocess_files_parallel(file_list, samples=SAMPLES, min_read_count=MIN_READ_COUNT)
        #
        # # ===== Create document-term matrix =====
        # # Read in preprocessed files.
        # file_list = [os.path.join('data/emerson_preprocessed', filename) for filename in
        #              os.listdir('data/emerson_preprocessed') if filename.endswith('.tsv') and filename.startswith('P')]
        # file_list.sort()
        # file_list = file_list[:SAMPLES]
        # # Create document-term matrix, save as .pkl since there are too many rows.
        # doc_term_matrix(file_list, filename_save=f'data/emerson_preprocessed/P0-P{SAMPLES}doc_term_matrix.pkl')

        # ===== Cluster sequences of document-term matrix =====
        document_term_matrix = pd.read_pickle(f'data/emerson_preprocessed/P0-P{SAMPLES}doc_term_matrix.pkl')
        document_term_matrix = document_term_matrix[:10000]

        # Clustering with chunks.
        if CLUSTERING_CHUNKS:
            datadir = 'data/chunks/'

            # Write the document-term matrix to chunks of specific size.
            write_dataframe_to_chunks(
                document_term_matrix.rename_axis(
                    'cdr3_amino_acid'
                ).reset_index()[['cdr3_amino_acid']],
                CLUSTERING_CHUNK_SIZE,
                f'{datadir}chunk',
                datadir,
            )

            # Calculate the total number of sequences in the dataset and save the chunk sized dataframes.
            files = []
            total_cdr3s = 0
            for file in os.listdir(datadir):
                df = pd.read_csv(datadir + file, sep='\t')
                files.append(df)
                total_cdr3s += len(df)
            print(f"Total number of sequences: {total_cdr3s}.")

            # Calculate the number of sequences to use for training.
            training_sample_size = round(1000 * (total_cdr3s / 5000))
            print(f"Training sample size: {training_sample_size}.")

            # Create metareportoire.
            training_sample = metareportoire(files, training_sample_size)

            # Calculate the maximum sequence length.
            max_seq_len = training_sample[0]['cdr3_amino_acid'].str.len().max()
            print(f"Max sequence length: {max_seq_len}.")

            # Initialize the clustering object.
            clustering = Clustering(faiss_training_data=training_sample[0]['cdr3_amino_acid'],
                                    fitting_data_size=total_cdr3s,
                                    max_sequence_size=max_seq_len,
                                    n_cpus=CLUSTERING_CPU)

            # Perform pre-clustering.
            for i in range(len(files)):
                print(f"Pre-clustering file: {str(i)}.")
                clustering.batch_precluster(files[i]['cdr3_amino_acid'])

            file_index = 0
            # Perform clustering.
            for result in clustering.batch_cluster():
                clusters_df = result.clusters_df
                summary = result.summary()
                process_cluster(document_term_matrix, clusters_df, summary, file_index=file_index,
                                tsv=CLUSTERING_USE_TSV)
                file_index += 1

            # Write to disk.
            if CLUSTERING_USE_TSV:
                all_cluster_assignments = pd.concat(
                    [pd.read_csv(f'data/emerson_preprocessed/cluster_assignments_{i}.tsv', sep='\t', index_col=0) for i
                     in
                     range(file_index)])
                all_cluster_assignments.to_csv('data/emerson_preprocessed/cluster_assignments.tsv', index=True,
                                               sep='\t')

                all_document_cluster_matrix = pd.concat(
                    [pd.read_csv(f'data/emerson_preprocessed/doc_term_matrix_{i}.tsv', sep='\t', index_col=0) for i in
                     range(file_index)])
                all_document_cluster_matrix.to_csv('data/emerson_preprocessed/doc_term_matrix.tsv', index=True,
                                                   sep='\t')
            else:
                all_cluster_assignments = pd.concat(
                    [pd.read_pickle(f'data/emerson_preprocessed/cluster_assignments_{i}.pkl') for i in
                     range(file_index)])
                all_cluster_assignments.to_pickle('data/emerson_preprocessed/cluster_assignments.pkl')

                all_document_cluster_matrix = pd.concat(
                    [pd.read_pickle(f'data/emerson_preprocessed/doc_term_matrix_{i}.pkl') for i in
                     range(file_index)])
                all_document_cluster_matrix.to_pickle('data/emerson_preprocessed/doc_term_matrix.pkl')

            # Clean up.
            clustering.batch_cleanup()
            # Remove the chunked dataframes
            for file in os.listdir('data/emerson_preprocessed/'):
                if file.startswith('cluster_assignments_') or file.startswith('doc_term_matrix_'):
                    os.remove(os.path.join('data/emerson_preprocessed/', file))
        else:

            result = cluster_single(document_term_matrix.rename_axis('cdr3_amino_acid').reset_index(),
                                    cdr3_col='cdr3_amino_acid', n_cpus=CLUSTERING_CPU)
            clusters_df = result.clusters_df
            summary = result.summary()

            process_cluster(document_term_matrix, clusters_df, summary)

            print(f'Found {len(summary)} clusters with an average size of {round(summary["size"].mean(), 2)}.')

        # ===== Topic Modelling =====
        # document_cluster_matrix = pd.read_csv('data/emerson_preprocessed/document_cluster_matrix_OG.tsv', sep='\t',
        #                                       index_col=0)
        #
        # document_cluster_matrix = pd.read_pickle('data/emerson_preprocessed/doc_term_matrix_OG.pkl')
        # print(document_cluster_matrix)
        # start = time.time()
        #
        # # 1. Preprocess your document-term matrix and convert it to gensim corpus format.
        # # Transpose the matrix so that rows are documents (samples) and columns are terms (sequences/clusters).
        # dtm_transposed = document_cluster_matrix.transpose()
        #
        # # Create a dictionary where the keys are sequences/clusters and the values are unique integer IDs.
        # dictionary = corpora.Dictionary([list(dtm_transposed.columns)])
        #
        #
        # corpus = []
        # for idx, row in dtm_transposed.iterrows():
        #     doc = []
        #     for seq, count in row.items():
        #         doc += [seq] * int(count)  # Add each sequence to the document `count` times.
        #     corpus.append(dictionary.doc2bow(doc))
        # print("Elapsed time: ", time.time() - start)
        #
        #
        # start = time.time()
        # # 2. Train an LDA model on your corpus.
        # lda_model = models.LdaMulticore(corpus, num_topics=10, id2word=dictionary, passes=15, workers=7,
        #                                 random_state=42)
        # print("Elapsed time: ", time.time() - start)
        #
        # # 3. Use the trained model to infer the topics in your documents.
        # doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
        #
        # print(doc_topics)
        #
        # topics_list = [[(topic_id, round(prob, 2)) for topic_id, prob in lda_model.get_document_topics(bow)] for bow in
        #                corpus]
        # topic_df = pd.DataFrame(topics_list, index=dtm_transposed.index)
        # topic_df.index = topic_df.index.astype(int)
        #
        # print(topic_df)
        #
        # # Load cluster assignments
        # # cluster_assignments = pd.read_csv('data/emerson_preprocessed/cluster_assignments.tsv', sep='\t', index_col=0)
        # # print(cluster_assignments)
        #
        # # Print each topic
        # for topic_id in range(lda_model.num_topics):
        #     print(f"Topic #{topic_id}")
        #
        #     # Get most contributing sequences
        #     for sequence, prob in lda_model.show_topic(topic_id):
        #         print(f"\tSequence: {sequence}, Probability: {prob:.2f}")
        #
        #         # # If sequence is a cluster, print its contents
        #         # if sequence in cluster_assignments['cluster_motif'].values:
        #         #     cluster_contents = cluster_assignments[cluster_assignments['cluster_motif'] == sequence][
        #         #         'junction_aa'].values
        #         #     print(f"\t\tCluster Contents: {', '.join(cluster_contents)}")
        #
        # metadata = pd.read_csv('data/emerson/metadata_merged.tsv', sep='\t')
        # metadata['sample_id'] = metadata['sample_id'].astype(int)
        #
        # # Join this DataFrame with your metadata DataFrame.
        # metadata_with_topics = metadata.join(topic_df, on='sample_id')
        # metadata_with_topics.to_csv('data/emerson_preprocessed/metadata_with_topics.tsv', sep='\t')
