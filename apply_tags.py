""" Sample docstring """
import pickle
import csv
import nltk
import sys
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from os.path import join
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn import datasets
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

flatten = lambda l: [item for sublist in l for item in sublist]

class Document(object):
    def __init__(self, id, title, content):
        self.Id = id
        self.Title = title
        self.Content = content

ITIS = datasets.load_iris()
LEMATIZER = nltk.stem.WordNetLemmatizer()
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def tokenize(text):
    for token in nltk.word_tokenize(text):
        token = token.lower()
        if token in STOPWORDS:
            continue
        yield LEMATIZER.lemmatize(token)

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct

START_INDEX = 0
def average_tfidf(path):
    reader = csv.DictReader(open(path, encoding="utf-8"))

    allDocuments = []
    for line in reader:
        allDocuments.append(Document(line["id"], line["title"], BeautifulSoup(line["content"], "lxml").get_text()))

    vectorizer = TfidfVectorizer(tokenizer=tokenize, decode_error='ignore')

    tdm = vectorizer.fit_transform (map( lambda i: "%s %s" % (i.Title, i.Content) ,allDocuments))
    features = vectorizer.get_feature_names()
    print("TDM contains %i terms and %i documents." % (len(features), tdm.shape[0]))
    physics_tags = defaultdict(set)
    with open(join("charts", "first-attempt.csv"), 'a') as f:
        if START_INDEX == 0:
            f.write("id,tags\n")
        for doc in allDocuments:
            if int(doc.Id) <= START_INDEX:
                continue
            for word in tokenize(doc.Content):
                if word in features:
                    if float(tdm[allDocuments.index(doc), features.index(word)]) > .3:
                        physics_tags[doc.Id].add(word)
            for word in tokenize(doc.Title):
                if word in features:
                    if float(tdm[allDocuments.index(doc), features.index(word)]) > .3:
                        physics_tags[doc.Id].add(word)
            f.write("%s,%s\n" % (doc.Id, " ".join(list(map(lambda t: str(str(t).encode(sys.stdout.encoding,errors="replace"))[2:][:-1], physics_tags[doc.Id])))))

for csv_name in ['test']:
    average_tfidf(join("data", csv_name + ".csv"))
