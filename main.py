#Importing required modules
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#NLTK Stopwords stored in a variable
stopWords = set(stopwords.words('english'))

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

#Function to find the cosine similarity of a two vectors
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

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


#Reading data from the csv using pandas
read_data = pd.read_csv('test.csv', encoding='latin')
all_documents =[]
for i in range(0,len(read_data)):
    all_documents.append(str(read_data['1'][i]))

#The text to compare
input_text = "dsgdf"


#Appending text to be compared to the doc vector
all_documents.append(input_text)

new_all =[[]]
for i in range(0,len(all_documents)):
    for j in (word_tokenize(all_documents[i])):
        if j not in stopWords:
            new_all[i].append(lemmet(j))
    new_all.append([])


for i in range(0,len(new_all)):
    new_all[i] = combine(new_all[i])


all_documents = new_all
print(all_documents[len(all_documents)-1])
        # all_documents[i][j] = str(lemmet(all_documents[i][j]))

#Converting the doc array to a vector
vectors = tfidf_from_all(all_documents).toarray()

#Finding the similarity of a given input with all the other fields in the csv
similarity = []
for i in range(0,len(vectors)-1):
    simm = cosine_similarity(vectors[len(vectors)-1],vectors[i])
    print("Similarity with idea ", i, " is ", simm)
    similarity.append(float(simm))

print(max(similarity))
print("Top Matches")
max_ind = index_of_max(similarity)
print(all_documents[max_ind])
similarity.remove(similarity[max_ind])
max_ind = index_of_max(similarity)
print(all_documents[max_ind])
similarity.remove(similarity[max_ind])
max_ind = index_of_max(similarity)
print(all_documents[max_ind])