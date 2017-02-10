from os.path import join
from bs4 import BeautifulSoup
import csv
import nltk

nltk.data.path.clear()
nltk.data.path.append("G:\\nltk_data")

class Document(object):
    def __init__(self, _id, title, content, tags):
        self._id = id
        self.title = title
        self.content = content
        self.tags = tags

LEMATIZER = nltk.stem.WordNetLemmatizer()
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def tokenize(text):
    for token in nltk.word_tokenize(text):
        token = token.lower()
        if token in STOPWORDS:
            continue
        yield LEMATIZER.lemmatize(token)

def calc_compaund_tags(path):
    reader = csv.DictReader(open(path, encoding="utf-8"))

    all_documetnts = []
    for line in reader:
        all_documetnts.append(Document(line["id"],
                                       line["title"],
                                       BeautifulSoup(line["content"], "lxml").get_text(),
                                       line["tags"]))
    questions_with_cmp_tag_count = 0
    for doc in all_documetnts:
        if any(map(lambda t: "-" in t, list(tokenize(doc.tags)))):
            questions_with_cmp_tag_count += 1

    res = questions_with_cmp_tag_count/len(all_documetnts)
    print("%s %f" % (path, res))

for csv_name in ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']:
    calc_compaund_tags(join("data", csv_name + ".csv"))
