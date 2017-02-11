import pickle
import csv
import nltk
import sys
import re
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
from os.path import join
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn import datasets
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

flatten = lambda l: [item for sublist in l for item in sublist]
get_lemma_names = lambda y: y.lemma_names()
get_hyponyms = lambda x: x.hyponyms()
get_hypernyms = lambda x: x.hypernyms()
get_hypo_and_hypernyms = lambda x: x.hypernyms()+x.hyponyms()
flat_list_map = lambda f, l: flatten(list(map(f, l)))

class Document(object):
    def __init__(self, id, title, content):
        self.Id = id
        self.Title = title
        self.Content = content

ITIS = datasets.load_iris()
LEMATIZER = nltk.stem.WordNetLemmatizer()
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
WORD_REGEX = re.compile("\\b[a-zA-Z][a-zA-Z-]*\\b")

def tokenize(text):
    for token in nltk.word_tokenize(text):
        token = token.lower()
        if token in STOPWORDS:
            continue
        match = WORD_REGEX.match(token)
        if match:
            token = match.group()
        yield LEMATIZER.lemmatize(token)

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct

def filer_nouns(word):
    return word[1] in ['NN', 'NNS', 'JJ', 'VBG']

def union_set(first_list, second_list):
    set(first_list+second_list)

START_INDEX = 0
MIN_TFIDF_VALUE = .3

def average_tfidf(path):
    reader = csv.DictReader(open(path, encoding="utf-8"))

    allDocuments = []
    for line in reader:
        allDocuments.append(Document(line["id"], line["title"], BeautifulSoup(line["content"], "lxml").get_text()))

    vectorizer = TfidfVectorizer(tokenizer=tokenize, decode_error='ignore')

    tdm = vectorizer.fit_transform(map(lambda i: "%s %s" %(i.Title, i.Content), allDocuments))
    features = vectorizer.get_feature_names()
    print("TDM contains %i terms and %i documents." % (len(features), tdm.shape[0]))
    physics_tags = defaultdict(set)
    with open(join("charts", "hypo-and-hypernyms-attempt.csv"), 'a') as f:
        if START_INDEX == 0:
            f.write("id,tags\n")
        for doc in allDocuments:
            if int(doc.Id) <= START_INDEX:
                continue
            for word_pair in filter(filer_nouns, pos_tag(list(tokenize(doc.Content)))):
                word = word_pair[0]
                if word in features:
                    if float(tdm[allDocuments.index(doc), features.index(word)]) > MIN_TFIDF_VALUE:
                        physics_tags[doc.Id].add(word)
                    else:
                        for w in filter(lambda w: tdm[allDocuments.index(doc), features.index(w)] > MIN_TFIDF_VALUE,
                                        filter(lambda w: w in features, flat_list_map(get_lemma_names, flat_list_map(get_hypo_and_hypernyms, wn.synsets(word))))):
                            physics_tags[doc.Id].add(w)

            for word_pair in filter(filer_nouns, pos_tag(list(tokenize(doc.Title)))):
                word = word_pair[0]
                if word in features:
                    if float(tdm[allDocuments.index(doc), features.index(word)]) > MIN_TFIDF_VALUE:
                        physics_tags[doc.Id].add(word)
                    else:
                        for w in filter(lambda w: tdm[allDocuments.index(doc), features.index(w)] > MIN_TFIDF_VALUE,
                                        filter(lambda w: w in features, flat_list_map(get_lemma_names, flat_list_map(get_hypo_and_hypernyms, wn.synsets(word))))):
                            physics_tags[doc.Id].add(w)
            f.write("%s,%s\n" % (doc.Id, " ".join(list(map(lambda t: str(str(t).encode(sys.stdout.encoding,errors="replace"))[2:][:-1], physics_tags[doc.Id])))))

for csv_name in ['biology']:
    average_tfidf(join("data", csv_name + ".csv"))
