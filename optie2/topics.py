from collections import defaultdict
from gensim import corpora, models, similarities
import numpy as np
import collections
import gensim.utils
from gensim.models import LdaMulticore


corpus2 = []
dictionary2 = []
docs = []
corpus = []
#files = ["data/new_P1_d0.csv", "data/new_P2_d15.csv", "data/new_Q1_d0.csv", "data/new_Q2_d15.csv", "data/new_S1_d0.csv", "data/new_S2_d15.csv"]
files = ["data/new_P1_d0.csv", "data/new_P1_d15.csv", "data/new_P2_d0.csv", "data/new_P2_d15.csv", "data/new_Q1_d0.csv", "data/new_Q1_d15.csv", "data/new_Q2_d0.csv", "data/new_Q2_d15.csv", "data/new_S1_d0.csv", "data/new_S1_d15.csv", "data/new_S2_d0.csv", "data/new_S2_d15.csv"]
dictionary = dict()
for file in files:
    with open(file, "r") as f:
        mens = []
        for line in f:
            aantal, woord = line.strip().split("\t")
            if int(aantal) > 4:
                for j in range(int(aantal)):
                    mens.append(woord)

        print(file)
        docs.append(mens)


dictionary2 = corpora.Dictionary(docs)
corpus2 = [dictionary2.doc2bow(text) for text in docs]
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

print('Number of unique tokens: %d' % len(dictionary2))
print('Number of documents: %d' % len(corpus2))

from gensim.models import LdaModel
num_topics = 6
chunksize = 2000
passes = 15
iterations =300
eval_every = None # Don't evaluate model perplexity, takes too much time.

temp = dictionary2[0]  # This is only to "load" the dictionary.
id2word = dictionary2.id2token

"""model = LdaMulticore(
    corpus=corpus2,
    id2word=id2word,
    num_topics=num_topics,
    passes=passes,
    iterations=iterations)"""

model = LdaModel(
    corpus=corpus2,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)
top_topics = model.top_topics(corpus2)
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)
from pprint import pprint
pprint(top_topics)

# which topic is dominant in each document
doc_topics = model.get_document_topics(corpus2)
for i, doc in enumerate(doc_topics):
    print("Document: {} Topic: {}".format(i, doc))


#tdidf = models.TfidfModel(corpus2)
#ldamodel = models.ldamodel.LdaModel(corpus2, num_topics=3, id2word = dictionary2, passes=20)

