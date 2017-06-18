from time import time
import string
import json
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
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
    output = open('tfidf_vocabulary.pkl', 'wb')
    pickle.dump(sklearn_tfidf.vocabulary_, output)
    output.close()
    # print((sklearn_tfidf.vocabulary_['your']))
    # json.dump(sklearn_tfidf.vocabulary_, open('vocabulary.json', mode='w'))
    # return sklearn_representation
    return [sklearn_representation,sklearn_tfidf.idf_]

#Function to combine a list into sentence
def combine(list):
    com=""
    for word in list:
        com = com+" "+word
    return com

#Function to find the maximum element in a given list
def index_of_max(list):
    max_value = max(list)
    max_index = list.index(max_value)
    return max_index


#NLTK Stopwords stored in a variable
stopWords = set(stopwords.words('english'))
for i in string.punctuation:
    stopWords.add(i)


#Reading data from the csv using pandas
read_data = pd.read_csv('eureka.csv', encoding='latin')
all_documents =[]
for i in range(0,len(read_data)):
    all_documents.append(str(read_data['1'][i]))


new_all =[[]]
for i in range(0,len(all_documents)):
    for j in (word_tokenize(all_documents[i])):
        if j not in stopWords:
            new_all[i].append(lemmet(j))
    new_all.append([])


for i in range(0,len(new_all)):
    new_all[i] = combine(new_all[i])


vectors = tfidf_from_all(all_documents)


#save vectors as python pickle
output = open('tfidf_idfs_data.pkl', 'wb')
pickle.dump(vectors, output)
output.close()


print("This took ",time()-initial_time)
