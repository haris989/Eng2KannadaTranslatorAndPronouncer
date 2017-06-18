from time import time
import pickle
import scipy.sparse as sp
import string
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
initial_time = time()

#Function to remove stopwords given a list of words
def remove_stopwords(list):
    final_list = []
    for word in list:
        if word not in stopWords:
            final_list.append(word)
    return final_list

#Function to perform lemmetization
def lemmet(word):
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    c = wordnet_lemmatizer.lemmatize(word)
    return c

#Function to get the tfid vector given a list of all vectors
def tfidf_from_all(all_docs):
    tokenize = lambda doc: doc.lower().split(" ")
    sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(all_docs)
    return sklearn_representation


def fit_one(doc):
    tokenize = lambda doc: doc.lower().split(" ")
    sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    return sklearn_tfidf.transform([doc])


#NLTK Stopwords stored in a variable
stopWords = set(stopwords.words('english'))
for i in string.punctuation:
    stopWords.add(i)

#Function to find the maximum element in a given list
def index_of_max(list):
    max_value = max(list)
    max_index = list.index(max_value)
    return max_index


#The text to compare
input_text = "What is the most efficient way to deep clone an object in JavaScript?"


#Appending text to be compared to the doc vector
pkl_file = open('tfidf_idfs_data.pkl', 'rb')

load =pickle.load(pkl_file)
vectors = load[0]
idfs = load[1]
pkl_file.close()

pkl_file = open('tfidf_vocabulary.pkl', 'rb')
vocabulary = pickle.load(pkl_file)
pkl_file.close()

#Converting the doc array to a vector

# print(vectors)


class MyVectorizer(TfidfVectorizer):
    # plug our pre-computed IDFs
    TfidfVectorizer.idf_ = idfs

# instantiate vectorizer
vectorizer = MyVectorizer(lowercase = False,
                          min_df = 2,
                          norm = 'l2',
                          smooth_idf = True)

# plug _tfidf._idf_diag


vectorizer._tfidf._idf_diag = sp.spdiags(idfs,
                                         diags = 0,
                                         m = len(idfs),
                                         n = len(idfs))


vectorizer.vocabulary_ = vocabulary

new_vector = vectorizer.transform([input_text])
# vectors.append(new_vector)
# print(new_vector.toarray()[0])
new_vector = new_vector.toarray()[0]
vectors = vectors.toarray()
# print(len(new_vector), len(vectors[1]))

# numpy.concatenate(vectors, new_vector)
# exit()
# vectors.append(new_vector.toarray()[0])
# exit()
# numpy.concatenate(vectors , new_vector)
#Finding the similarity of a given input with all the other fields in the csv
time_cosine = time()
similarity = []
for i in range(0,len(vectors)-1):
    simm = cosine_similarity([new_vector], [vectors[i]])
    # print("Similarity with idea ", i, " is ", simm)
    similarity.append(float(simm))
print("cosine similarity took ",time()-initial_time)
# print(max(similarity))


print("Top Matches")
max_ind = index_of_max(similarity)
print(max_ind)
# print(all_documents[max_ind])
similarity[max_ind]=0
max_ind = index_of_max(similarity)
print(max_ind)
# print(all_documents[max_ind])
similarity[max_ind]=0
max_ind = index_of_max(similarity)
print(max_ind)
# print(all_documents[max_ind])

print("This took ",time()-initial_time)