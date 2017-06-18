from sklearn.feature_extraction.text import TfidfVectorizer

all_docs =["what is your name","what his mame", "What is the spell of name"]
tokenize = lambda doc: doc.lower().split(" ")
sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_docs)


print(sklearn_tfidf.idf_)
exit()
print(sklearn_tfidf.get_feature_names())
print(sklearn_representation)
# print(sklearn_tfidf.get_feature_names())
print("sdfdsn \n \n")

sklearn_representation2 = sklearn_tfidf.transform(["his name whatt is of  chu"])
print(sklearn_representation2,"sdfdsn \n \n")
# print(sklearn_representation)
# print(sklearn_tfidf.get_feature_names())
# corpus_tf_idf = vect.transform(corpus)
# doc = "my name is khan"
# doc_tfidf = vect.transform([doc])