import math
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

stopWords = set(stopwords.words('english'))

def remove_stopwords(list):
    final_list=[]
    for word in list:
        if word not in stopWords:
            final_list.append(word)
    return final_list

def lemmet(word):
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    c = wordnet_lemmatizer.lemmatize(word)
    return c


def tfidf_from_all(all_docs):
    tokenize = lambda doc: doc.lower().split(" ")
    sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(all_docs)
    return sklearn_representation


def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

def combine(list):
    com=[]
    for word in list:
        com = com+word
    return com

def index_of_max(list):
    max_value = max(list)
    max_index = list.index(max_value)
    return max_index



read_data = pd.read_csv('test.csv', encoding='latin')
all_documents =[]
for i in range(0,len(read_data)):
    all_documents.append(str(read_data['1'][i]) + " "+ str(read_data['2'][i])+" " + str(read_data['3'][i]))
input_text = "how find length of a given list in python"
all_documents.append(input_text)
vectors = tfidf_from_all(all_documents).toarray()
print(len(vectors))
similarity = []
for i in range(0,len(vectors)-1):
    simm = cosine_similarity(vectors[len(vectors)-1],vectors[i])
    similarity.append(float(simm))
    #print("Similarity with idea ",i," is ",simm)

print("Top Matches")
max_ind = index_of_max(similarity)
print(all_documents[max_ind])
similarity.remove(similarity[max_ind])
max_ind = index_of_max(similarity)
print(all_documents[max_ind])
similarity.remove(similarity[max_ind])
max_ind = index_of_max(similarity)
print(all_documents[max_ind])