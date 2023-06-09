import os
import time
import random
import requests

from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def read_file(path: str, person: str, timestamps: list[str], saveAll=True) -> dict:
    """
    Reads in TSV files for a given person and timestamps, and saves them to pickle format if they weren't already stored in pickles.

    :param path: str, path to the directory containing the files.
    :param person: str, person identifier (e.g., 'P1').
    :param timestamps: list of str, list of day identifiers (e.g., ['d0', 'd15']).
    :param saveAll: bool, optional, whether to save all columns or just the CDR3 amino acid sequence. Defaults to True.

    :return: dict, dictionary of pandas dataframes, keyed by person-day (e.g., {'P1_d0': dataframe, 'P1_d15': dataframe, ...}).

    :raises: Exception if there's an error reading or saving a file.
    """
    dataframes = {}
    for timestamp in timestamps:
        filename = f"{person}_{timestamp}.tsv"
        pickled_filename = f"{person}_{timestamp}.pkl"
        if os.path.isfile(os.path.join(path, pickled_filename)):
            # If pickled file exists, read from it.
            dataframes[f"{person}_{timestamp}"] = pd.read_pickle(os.path.join(path, pickled_filename)) if saveAll else \
                pd.read_pickle(os.path.join(path, pickled_filename))['CDR3.amino.acid.sequence']
            print(f"Read {os.path.join(path, pickled_filename)} from disk.")
        else:
            # Otherwise, read TSV file and save to pickled file.
            filepath = os.path.join(path, filename)
            df = read_tsv(filepath, delimiter='\t')
            save_df(df, os.path.join(path, pickled_filename))
            dataframes[f"{person}_{timestamp}"] = df if saveAll else df['CDR3.amino.acid.sequence']
    return dataframes


def read_files_dict(path: str, persons: list[str], timestamps: list[str], saveAll=True) -> dict:
    """
    Reads in TSV files for given persons and timestamps, and saves them to pickle format if they weren't already stored in pickles.

    :param path: str, path to the directory containing the files.
    :param persons: list of str, list of person identifiers (e.g., ['P1', 'P2', 'P3']).
    :param timestamps: list of str, list of day identifiers (e.g., ['d0', 'd15']).
    :param saveAll: bool, optional, whether to save all columns or just the CDR3 amino acid sequence. Defaults to True.

    :return: dict, dictionary of pandas dataframes, keyed by person-day (e.g., {'P1_d0': dataframe, 'P1_d15': dataframe, ...}).

    :raises: Exception if there's an error reading or saving a file.
    """
    dataframes = {}
    for person in persons:
        for timestamp in timestamps:
            filename = f"{person}_{timestamp}.tsv"
            pickled_filename = f"{person}_{timestamp}.pkl"
            if os.path.isfile(path + pickled_filename):
                # If pickled file exists, read from it.
                dataframes[f"{person}_{timestamp}"] = pd.read_pickle(path + pickled_filename) if saveAll else \
                    pd.read_pickle(path + pickled_filename)['CDR3.amino.acid.sequence']
                print(f"Read {path + pickled_filename} from disk.")
            else:
                # Otherwise, read TSV file and save to pickled file.
                filepath = os.path.join(path, filename)
                df = read_tsv(filepath, delimiter='\t')
                save_df(df, path + pickled_filename)
                dataframes[f"{person}_{timestamp}"] = df if saveAll else df['CDR3.amino.acid.sequence']
    return dataframes


def read_files_df(path: str, persons: list[str], timestamps: list[str], saveAll=True, num_lines=None) -> pd.DataFrame:
    """
    Reads in TSV files for given persons and timestamps, and returns a combined DataFrame.

    :param path: str, path to the directory containing the files.
    :param persons: list of str, list of person identifiers (e.g., ['P1', 'P2', 'P3']).
    :param timestamps: list of str, list of day identifiers (e.g., ['d0', 'd15']).
    :param saveAll: bool, optional, whether to save all columns or just the CDR3 amino acid sequence. Defaults to True.
    :param num_lines: int, optional, number of lines to read from each file. If not specified, reads all lines.

    :return: pd.DataFrame, combined DataFrame containing TCR sequences for all persons and timestamps.

    :raises: Exception if there's an error reading or saving a file.
    """
    df_list = []
    for person in persons:
        for timestamp in timestamps:
            filename = f"{person}_{timestamp}.tsv"
            pickled_filename = f"{person}_{timestamp}.pkl"
            if os.path.isfile(path + pickled_filename):
                # If pickled file exists, read from it.
                df = pd.read_pickle(path + pickled_filename)
                print(f"Read {path + pickled_filename} from disk.")
            else:
                # Otherwise, read TSV file and save to pickled file.
                filepath = os.path.join(path, filename)
                df = pd.read_csv(filepath, sep='\t')
                save_df(df, path + pickled_filename)
            if not saveAll:
                df = df[['CDR3.amino.acid.sequence']]
            df['person'] = person
            df['timestamp'] = timestamp
            df_list.append(df.head(num_lines))
    df = pd.concat(df_list, ignore_index=True)
    return df


def timer_decorator(func):
    """
    Decorator function that times the execution of another function.

    :param func: The function to be timed.
    :return: The result returned by the function.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function that adds timing functionality to the decorated function.

        :param args: Positional arguments to be passed to the decorated function.
        :param kwargs: Keyword arguments to be passed to the decorated function.
        :return: The result returned by the decorated function.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result

    return wrapper


def visualize_features(features: pd.DataFrame):
    # Reference - https://svalkiers.github.io/clusTCR/docs/analyzing/features.html

    # Plot entropy of each cluster.
    fig, ax = plt.subplots()
    ax.bar(features.index, features['h'])
    plot_histogram(
        ax, 'Cluster', 'Entropy', 'Entropy for Clusters'
    )
    # Plot the number of sequences in each cluster.
    fig, ax = plt.subplots()
    ax.bar(features.index, features['size'])
    plot_histogram(
        ax, 'Cluster', 'Number of Sequences', 'Size of Clusters'
    )
    # Length of sequences in the cluster
    fig, ax = plt.subplots()
    ax.hist(features['length'], bins=range(25))
    plot_histogram(
        ax,
        'CDR3 Sequence Length',
        'Frequency',
        'Length of CDR3 Sequences in Clusters',
    )
    # Plot the average and variance basicity of each cluster.
    fig, ax = plt.subplots()
    ax.scatter(features['basicity_avg'], features['basicity_var'])
    plot_histogram(
        ax, 'Average Basicity', 'Variance in Basicity', 'Basicity in Clusters'
    )


def plot_histogram(ax, arg1, arg2, arg3):
    ax.set_xlabel(arg1)
    ax.set_ylabel(arg2)
    ax.set_title(arg3)
    plt.show()


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


def write_ndarray_to_chunks(arr, chunk_size, filename_prefix, output_dir):
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

    num_chunks = len(arr) // chunk_size + 1

    for i, chunk_start in enumerate(range(0, len(arr), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(arr))
        chunk = arr[chunk_start:chunk_end]

        filename = f"{filename_prefix}_chunk{i + 1}.txt"
        np.savetxt(filename, chunk, delimiter='\t', fmt='%s')
        print(f"Chunk {i + 1} saved to {filename}")