import random
from typing import Tuple

from clustcr import read_cdr3, Clustering
import os
import pandas as pd
import utils





@utils.timer_decorator
def lol(a):
    # First, we define the path to the data directory
    datadir = 'data/chunks/'

    # Now we count the number of sequences that are present in the data set
    # We use os.listdir() to list all the files in the specified directory
    files = []
    total_cdr3s = 0
    for file in os.listdir(datadir):
        df = pd.read_csv(datadir + file, sep='\t')
        files.append(df)
        total_cdr3s += len(df)
        # total_cdr3s += len(read_cdr3(datadir + file,
        #                              data_format='tcrex'))
    print(f"Total number of CDR3s: {total_cdr3s}")

    training_sample_size = round(1000 * (total_cdr3s / 5000))
    # training_sample = metarepertoire(directory=datadir,
    #                                  data_format='tcrex',
    #                                  n_sequences=training_sample_size)

    training_sample = metareportoire(files, training_sample_size)


    max_seq_len = training_sample[0]['cdr3_amino_acid'].str.len().max()
    print(f"Max sequence length: {max_seq_len})")
    print(f"Training sample size: {training_sample_size}")

    clustering = Clustering(faiss_training_data=training_sample[0]['cdr3_amino_acid'],
                            fitting_data_size=total_cdr3s,
                            max_sequence_size=max_seq_len,
                            n_cpus=a)
    print("Clustering initialized.")

    for i in range(len(files) - 1):
        print(f"Preclustering... file: {str(i)}...")
        # Load your data
        # data = read_cdr3(file=os.path.join(datadir, files[i]),
        #                  data_format='tcrex')

        clustering.batch_precluster(files[i]['cdr3_amino_acid'])



    for cluster in clustering.batch_cluster():
        print(cluster.clusters_df)

    clustering.batch_cleanup()





if __name__ == '__main__':
    # df = pd.read_pickle('data/emerson_preprocessed/P0-P100doc_term_matrix.pkl') \
    #     .rename_axis('cdr3_amino_acid').reset_index()[['cdr3_amino_acid']]
    #
    # write_dataframe_to_chunks(df, 100000, 'data/chunks/chunk')

    lol(a=7)
    lol(a='7')
