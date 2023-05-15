# Preprocess the sequences.
for key, value in dict_df.items():
    # Remove sequences containing unresolved amino acids.
    dict_df[key] = value[~value['CDR3.amino.acid.sequence'].str.contains('X')]
    # Remove duplicates.
    value['CDR3.amino.acid.sequence'].drop_duplicates(inplace=True)
    # Remove sequences with a read count under a specified threshold.
    value = value.loc[value['Read.count'] > 0]
    value = value.head(20000)
    dict_df[key] = value

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

p1_d0 = df['P1_d0']

# Reference - https://svalkiers.github.io/clusTCR/
clustering = Clustering(method='two-step', n_cpus='all', faiss_cluster_size=5000, mcl_params=[1.2, 2])

results = clustering.fit(p1_d0[:1000])
# Include CDR3 alpha chain.

# Retrieve dataframe containing clusters.
# results.clusters_df

# Retrieve CDR3's in each cluster.
# print(results.cluster_contents())

# Retrieve features of clustering.
features = results.compute_features(compute_pgen=True)

visualize_features(features)


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

    # persons = ['P1', 'P2', 'Q1', 'Q2', 'S1', 'S2']
    # days = ['d0', 'd15']
    # # df = read_files_df('data/Pogorelyy_YF/', persons, days, saveAll=True, num_lines=10000)
    #
    # # df = preprocess_tcr_sequences(df, unresolved='X', min_read_count=1)

    # # Calculate the total number of sequences in the dataset.
    # total_sequences = df.shape[0]
    # # Cluster size.
    # faiss_cluster_size = 5000
    # # Calculate recommended sample size.
    # training_sample_size = round(1000 * (total_sequences / faiss_cluster_size))
    #
    # # Create metareportoire.
    # meta = df[['CDR3.amino.acid.sequence']].copy()
    # # Remove duplicates.
    # meta.drop_duplicates(inplace=True)
    # # Randomly sample the metareportoire.
    # if len(meta) > training_sample_size:
    #     meta = meta.sample(training_sample_size)
    # # Compute the maximum sequence length in the metareportoire.
    # max_seq_len = meta['CDR3.amino.acid.sequence'].str.len().max()
    #
    # # Create a dictionary for chunks.
    # if not os.path.exists('data/Pogorelyy_YF/chunks'):
    #     os.makedirs('data/Pogorelyy_YF/chunks')
    # # Clear .pkl files in 'chunks' directory.
    # [os.remove(os.path.join(dirpath, file)) for dirpath, dirnames, filenames in os.walk('data/Pogorelyy_YF/chunks') for
    #  file
    #  in filenames if file.endswith('.pkl')
    #  ]
    #
    # time.sleep(1)
    #
    # # Clear clustcr_batch directories.
    # for dir in os.listdir('./'):
    #     if dir.startswith('clustcr_batch') and os.path.isdir(os.path.join('./', dir)):
    #         shutil.rmtree(os.path.join('./', dir))
    #
    # time.sleep(1)
    #
    # chunk_size = 100000
    # num_chunks = (df.shape[0] + chunk_size - 1) // chunk_size
    # # Save chunks to disk.
    # for i in range(num_chunks):
    #     start = i * chunk_size
    #     end = min(start + chunk_size, len(df))
    #     save_df(df.iloc[start:end], f'data/Pogorelyy_YF/chunks/chunk_{i}.pkl')
    #
    # # Remove to free memory.
    # # del df
    #
    # # clustering = Clustering(n_cpus='all', method='two-step', mcl_params=[1.2, 2])
    # # result = clustering.fit(df['CDR3.amino.acid.sequence'])
    # # result.clusters_df
    #
    # print(
    #     f'Perform clustering with a faiss_cluster_size of {faiss_cluster_size} and a training_sample_size of {training_sample_size} with a total of {total_sequences} sequences.')
    # # Create clustering object.
    # clustering = Clustering(faiss_training_data=meta['CDR3.amino.acid.sequence'], fitting_data_size=total_sequences,
    #                         max_sequence_size=max_seq_len, n_cpus='all', method='two-step',
    #                         faiss_cluster_size=faiss_cluster_size, mcl_params=[1.2, 2])
    #
    # print("Batch preclustering...")
    # filenames = sorted(file for file in os.listdir('data/Pogorelyy_YF/chunks') if file.endswith('.pkl'))
    # for file in filenames:
    #     f = pd.read_pickle(os.path.join('data/Pogorelyy_YF/chunks', file))
    #     clustering.batch_precluster(f['CDR3.amino.acid.sequence'])
    #
    # print("Batch clustering...")
    # clusters = list(clustering.batch_cluster())
    # # Replace TCR sequences with their corresponding cluster labels.
    # for cluster_obj in clusters:
    #     cluster_df = cluster_obj.clusters_df
    #     cluster_dict = dict(zip(cluster_df["junction_aa"], cluster_df["cluster"]))
    #     df["CDR3.amino.acid.sequence"] = df["CDR3.amino.acid.sequence"].map(cluster_dict)
    #
    # print('clustring done')
    # # Group the TCR sequences (now represented by cluster labels) by individual and timestamp.
    # grouped = df.groupby(['person', 'timestamp'])['CDR3.amino.acid.sequence'].apply(
    #     lambda x: ' '.join(map(str, x))).reset_index()
    #
    # # Create a CountVectorizer to vectorize the cluster labels.
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(grouped['CDR3.amino.acid.sequence'])
    #
    # # Perform topic modeling using Latent Dirichlet Allocation (LDA).
    # n_topics = 5  # Number of topics to extract
    # lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    # topic_distributions = lda.fit_transform(X)
    #
    # # Assign the topic withrrrr the highest probability to each group.
    # grouped['topic'] = topic_distributions.argmax(axis=1)
    #
    # # Analyze the results.
    # topic_words = {}
    # for topic_idx, topic in enumerate(lda.components_):
    #     words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]]
    #     topic_words[f"Topic {topic_idx}"] = words
    #
    # print(topic_words)

    def process_files():
        filenames = ['P1_d0', 'P1_d15', 'P2_d0', 'P2_d15', 'Q1_d0', 'Q1_d15', 'Q2_d0', 'Q2_d15', 'S1_d0', 'S1_d15',
                     'S2_d0', 'S2_d15']

        pool = multiprocessing.Pool()

        # use partial to fix the filename_save argument to the preprocess_file function
        preprocess_func = partial(preprocess_file, min_read_count=10)

        results = []

        for filename in filenames:
            filename_with_path = 'data/Pogorelyy_YF/{}.pkl'.format(filename)

            # skip files that have already been processed
            if os.path.exists('pre_processed_{}.csv'.format(filename)):
                continue

            result = pool.apply_async(preprocess_func,
                                      args=(filename_with_path, 'pre_processed_{}.csv'.format(filename)))
            results.append(result)

        pool.close()
        pool.join()

        # check for exceptions
        for result in results:
            try:
                result.get()
            except Exception as e:
                print(f"A task failed with exception: {type(e).__name__}, {e.args}")

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
