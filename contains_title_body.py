"""This module presents graphics about whether a stemmed tag is contained in the title or body of the question"""
import csv
import re
from os.path import join
from collections import defaultdict
from nltk.stem.lancaster import LancasterStemmer
import matplotlib.pyplot as plt


def save_contains_charts(csv_path, csv_name):
    stemmer = LancasterStemmer()
    print("Working on %s" % csv_path)
    reader = csv.DictReader(open(csv_path, encoding="utf8"))
    contained_title = defaultdict(int)
    contained_body = defaultdict(int)
    contained_both = defaultdict(int)
    tags_num = 0
    remove_html_and_eol_regex = re.compile(r"<[^<]+?>|(\\r)?\\n", re.IGNORECASE)
    find_words_regex = re.compile(r'\w+')
    for post in reader:
        contents_words_stripped = remove_html_and_eol_regex.sub("", str(str(post["content"]).encode('utf8')))
        contents_words = find_words_regex.findall(contents_words_stripped)
        stemmed_contents = [stemmed_word for stemmed_word in map(stemmer.stem, contents_words)]

        title_words_stripped = remove_html_and_eol_regex.sub("", str(str(post["title"]).encode('utf8')))
        title_words = find_words_regex.findall(title_words_stripped)
        stemmed_title = [stemmed_word for stemmed_word in map(stemmer.stem, title_words)]

        stemmed_tags = [stemmed_tag for stemmed_tag in map(stemmer.stem, post["tags"].split(" "))]
        for tag in stemmed_tags:
            tags_num += 1
            contains_body = tag in stemmed_contents
            contains_title = tag in stemmed_title
            contained_body[contains_body] += 1
            contained_title[contains_title] += 1
            contained_both[contains_body and contains_title] += 1
    for key in contained_title:
        contained_body[key] = contained_body[key] / tags_num * 100
        contained_title[key] = contained_title[key] / tags_num * 100
        contained_both[key] = contained_both[key] / tags_num * 100

    data = {'title': contained_title[True],
            'body': contained_body[True],
            'both': contained_both[True]}
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(data)), data.values(), align='center')
    plt.xticks(range(len(data)), data.keys())
    plt.xlabel("Contained in ")
    plt.ylabel("Percent of words")
    plt.suptitle(csv_name)
    plt.savefig(join("charts", "contains", csv_name + ".png"))

for csv_name in ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']:
    save_contains_charts(join("data", csv_name + ".csv"), csv_name)
