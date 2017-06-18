import gensim
import numpy
from time import time
initial_time = time()

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]


sentences = [i.lower().split() for i in documents]

model = gensim.models.Word2Vec(sentences, min_count=1, size=10,iter=25)

vocab = list(model.wv.vocab)

modelvect = model[vocab]

index = gensim.similarities.MatrixSimilarity(corpus=modelvect, num_features=10)
print(index)
X = numpy.array([gensim.matutils.unitvec(i) for i in modelvect])

print(index[X])

print("This took ",time()-initial_time)