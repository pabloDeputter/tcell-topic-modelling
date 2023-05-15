import sys
from collections import defaultdict
from gensim import corpora, models, similarities
import numpy as np
import collections
import gensim.utils
from gensim.models import LdaMulticore
import cython

def read_data(filenames: list[str]):
    corpus : list[list[tuple[int, int]]] = [] # list van lijsten van tuples (id, aantal)
    dictionary = dict() # key = id, value = woord
    reverse_dictionary = dict() # key = woord, value = id
    id_teller = 0
    for filename in filenames:
        mens : list[tuple[int, int]] = []
        with open(filename, "r") as f:
            for line in f:
                # if the line ends on a tab, skip
                fields = line.strip().split("\t")
                if len(fields) != 2:
                    continue
                aantal, woord = fields
                if int(aantal) > 2:
                    if woord not in reverse_dictionary:
                        reverse_dictionary[woord] = id_teller
                        dictionary[id_teller] = woord
                        id_teller += 1
                    mens.append((reverse_dictionary[woord], int(aantal)))
        corpus.append(mens)
    print("Number of unique tokens: %d" % len(dictionary))
    return corpus, dictionary

"""
corpus2 = []
dictionary2 = []
docs = []
corpus = []
#files = ["data/new_P1_d0.csv", "data/new_P2_d15.csv", "data/new_Q1_d0.csv", "data/new_Q2_d15.csv", "data/new_S1_d0.csv", "data/new_S2_d15.csv"]
files = ["data/new_P1_d0.csv", "data/new_P1_d15.csv", "data/new_P2_d0.csv", "data/new_P2_d15.csv", "data/new_Q1_d0.csv", "data/new_Q1_d15.csv", "data/new_Q2_d0.csv", "data/new_Q2_d15.csv", "data/new_S1_d0.csv", "data/new_S1_d15.csv", "data/new_S2_d0.csv", "data/new_S2_d15.csv"]
for file in files:
    with open(file, "r") as f:
        mens = []
        mens2 = []
        for line in f:
            aantal, woord = line.strip().split("\t")
            if int(aantal) > 2:
                for j in range(int(aantal)):
                    mens.append(woord)

                    

        print(file)
        docs.append(mens)
"""


#dictionary2 = corpora.Dictionary(docs)
#corpus2 = [dictionary2.doc2bow(text) for text in docs]
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

#print('Number of unique tokens: %d' % len(dictionary2))
#print('Number of documents: %d' % len(corpus2))


def doLDA(corpus: list[list[tuple[int, int]]], dictionary : dict[int, str], num_topics: int):
    from gensim.models import LdaModel
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
    chunksize = 200000
    passes = 10
    iterations =350
    eval_every = None # Don't evaluate model perplexity, takes too much time.
    #temp = dictionary2[0]  # This is only to "load" the dictionary.
    #id2word = dictionary2.id2token
    id2word = dictionary

    """model = LdaMulticore(
    corpus=corpus2,
    id2word=id2word,
    num_topics=num_topics,
    passes=passes,
    iterations=iterations)"""

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    top_topics = model.top_topics(corpus)
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    from pprint import pprint
    pprint(top_topics)

    # which topic is dominant in each document
    doc_topics = model.get_document_topics(corpus)
    return doc_topics

def getNewFilenames(directory: str):
    # return filenames that start witn new_ and end with .csv
    import glob
    out = glob.glob(directory + "/new_*.tsv")
    print(len(out))
    return glob.glob(directory + "/new_*.tsv")
#tdidf = models.TfidfModel(corpus2)
#ldamodel = models.ldamodel.LdaModel(corpus2, num_topics=3, id2word = dictionary2, passes=20)


directory = "data/processed_files"
filenames = getNewFilenames(directory)
corpus, dictionary = read_data(filenames)
print("hier")
doc_topics = doLDA(corpus, dictionary, 50)
for i in range(len(doc_topics)):
    print("{} ::: Topic: {}".format(filenames[i].split("/")[-1], doc_topics[i]))




