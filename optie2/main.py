from collections import defaultdict
from gensim import corpora, models, similarities
import numpy as np
import collections
import gensim.utils

VacDoc = collections.namedtuple('VacDoc', 'words tags split vacstatus')


corpus2 = []
dictionary2 = []
docs = []
#files = ["data/new_P1_d0.csv", "data/new_P2_d15.csv", "data/new_Q1_d0.csv", "data/new_Q2_d15.csv", "data/new_S1_d0.csv", "data/new_S2_d15.csv"]
files = ["data/new_P1_d0.csv", "data/new_P1_d15.csv", "data/new_P2_d0.csv", "data/new_P2_d15.csv", "data/new_Q1_d0.csv", "data/new_Q1_d15.csv", "data/new_Q2_d0.csv", "data/new_Q2_d15.csv", "data/new_S1_d0.csv", "data/new_S1_d15.csv", "data/new_S2_d0.csv", "data/new_S2_d15.csv"]
for i in range(len(files)):
    with open(files[i], "r") as f:
        mens = []
        for line in f:
            aantal, woord = line.strip().split("\t")
            if int(aantal) > 4:
                for j in range(int(aantal)):
                    mens.append(woord)
        print(i)
        if files[i].split("_")[2] == "d0":
            docs.append(VacDoc(mens, [i], 'train', 0))
        else:
            docs.append(VacDoc(mens, [i], 'train', 1))

test_docs = docs[0:2]
train_docs = docs[2:]

import multiprocessing
from collections import OrderedDict

import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

from gensim.models.doc2vec import Doc2Vec

common_kwargs = dict(
        vector_size=100, epochs=20, min_count=2,
        sample=0, workers=multiprocessing.cpu_count(), negative=5, hs=0,
    )
simple_models = [
    # PV-DBOW plain
    Doc2Vec(dm=0, **common_kwargs),
    # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
    Doc2Vec(dm=1, window=10, alpha=0.05, comment='alpha=0.05', **common_kwargs),
    # PV-DM w/ concatenation - big, slow, experimental mode
    # window=5 (both sides) approximates paper's apparent 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, window=5, **common_kwargs),
    ]

for model in simple_models:
    model.build_vocab(docs)
    print(f"{model} vocabulary scanned & state initialized")

models_by_name = OrderedDict((str(model), model) for model in simple_models)
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])


import numpy as np
import statsmodels.api as sm
from random import sample

def logistic_predictor_from_data(train_targets, train_regressors):
    """Fit a statsmodel logistic predictor on supplied data"""
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    #print(predictor.summary())
    return predictor

def error_rate_for_model(test_model, train_set, test_set):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets = [doc.vacstatus for doc in train_set]
    train_regressors = [test_model.dv[doc.tags[0]] for doc in train_set]
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_regressors = [test_model.dv[doc.tags[0]] for doc in test_set]
    test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.vacstatus for doc in test_set])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)

from collections import defaultdict
error_rates = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved

from random import shuffle
shuffled_alldocs = docs[:]
shuffle(shuffled_alldocs)

for model in simple_models:
    print(f"Training {model}")
    model.train(shuffled_alldocs, total_examples=len(shuffled_alldocs), epochs=model.epochs)

    print(f"\nEvaluating {model}")
    #err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
    #error_rates[str(model)] = err_rate
    #print("\n%f %s\n" % (err_rate, model))

for model in [models_by_name['dbow+dmm'], models_by_name['dbow+dmc']]:
    print(f"\nEvaluating {model}")
    err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
    error_rates[str(model)] = err_rate
    # pritn all predictions of the model

    print(f"\n{err_rate} {model}\n")



print("Err_rate Model")
for rate, name in sorted((rate, name) for name, rate in error_rates.items()):
    print(f"{rate} {name}")

"""
dictionary2 = corpora.Dictionary(docs)
corpus2 = [dictionary2.doc2bow(text) for text in docs]
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models import LdaModel
num_topics = 7
chunksize = 20
passes = 100
iterations = 400
eval_every = 10  # Don't evaluate model perplexity, takes too much time.

temp = dictionary2[0]  # This is only to "load" the dictionary.
id2word = dictionary2.id2token

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
"""

