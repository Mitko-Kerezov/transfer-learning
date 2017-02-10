from os.path import join
from re import compile as re_compile
from csv import DictReader
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from gensim import corpora, models
from bs4 import BeautifulSoup

def get_ldamodel(doc_set):
    texts = []
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()

    for i in doc_set:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return  models.ldamodel.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=20)


def run_lda(csv_path):
    pattern = re_compile('"(.*?)"')
    print("Working on %s" % csv_path)
    reader = DictReader(open(csv_path, encoding="utf-8"))
    for post in reader:
        doc_set = BeautifulSoup(post["content"], "lxml").get_text().split()
        words = pattern.findall(get_ldamodel(doc_set).show_topics(num_topics=1, num_words=5)[0][1])
        print(post["id"])
        real_tags = post["tags"].split()
        for word in filter(lambda w: w in real_tags, words):
            print(word)


for csv_name in ['biology']:
    run_lda(join("data", csv_name + ".csv"))
