""" abs """
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
from os.path import join
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn import datasets
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.data.path.clear()
nltk.data.path.append("G:\\nltk_data")
flatten = lambda l: [item for sublist in l for item in sublist]

class Document(object):
    def __init__(self, id, title, content, tags):
        self.Id = id
        self.Title = title
        self.Content = content
        self.Tags = tags

ITIS = datasets.load_iris()
LEMATIZER = nltk.stem.WordNetLemmatizer()
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def tokenize(text):
    for token in nltk.word_tokenize(text):
        token = token.lower()
        if token in STOPWORDS or not token.isalpha():
            continue
        yield LEMATIZER.lemmatize(token)

def average_tfidf(path):
    reader = csv.DictReader(open(path, encoding="utf8"))

    allContents = []
    allTags = []

    allDocumetnts = []
    for line in reader:
        allDocumetnts.append(Document(line["id"], line["title"], BeautifulSoup(line["content"], "lxml").get_text(), line["tags"]))

    vectorizer = TfidfVectorizer(tokenizer=tokenize, decode_error='ignore')

    tdm = vectorizer.fit_transform (map( lambda i: "%s %s" % (i.Title, i.Content)  ,allDocumetnts))
    features = vectorizer.get_feature_names()
    print("TDM contains %i terms and %i documents." % (len(features), tdm.shape[0]))
    SUM = 0
    COUNT = 0
    SUM_TITLE = 0
    COUNT_TITLE = 0
    for doc in allDocumetnts:
        #print()
        #print("-------------------------------")
        if int(doc.Id)%100 == 0:
            print("Document %s" % doc.Id)
        for word in tokenize(doc.Content):
            if word in features:
                #print("%s %f %s" % (word, tdm[allDocumetnts.index(doc), features.index(word)], "True" if word in doc.Tags else ""))
                if(word in doc.Tags):
                    SUM += tdm[allDocumetnts.index(doc), features.index(word)]
                    COUNT += 1
        for word in tokenize(doc.Title):
            if word in features:
                #print("%s %f %s" % (word, tdm[allDocumetnts.index(doc), features.index(word)], "True" if word in doc.Tags else ""))
                if(word in doc.Tags):
                    SUM_TITLE += tdm[allDocumetnts.index(doc), features.index(word)]
                    COUNT_TITLE += 1

    print("AVERAGE CONTENT %f " % (SUM/COUNT))
    print("AVERAGE TITLE %f " % (SUM_TITLE/COUNT_TITLE))
    return (SUM/COUNT),(SUM_TITLE/COUNT_TITLE)

total_title = 0
total_content = 0
for csv_name in ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']:
    print("EI TUI CHETEM BRAT %s" % csv_name)
    avr_count, avr_count_title = average_tfidf(join("transfer-learning", "data", csv_name + ".csv"))
    total_title+=avr_count_title
    total_content+=avr_count
