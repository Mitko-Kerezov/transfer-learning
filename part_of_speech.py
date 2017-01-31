"""This module does presents graphics about what part of speech each of the tags represent."""
import csv
from os.path import join
from collections import defaultdict
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


def print_tags(csv_path, csv_name):
    print("Working on %s" % csv_path)
    wnl = WordNetLemmatizer()
    reader = csv.DictReader(open(csv_path, encoding="utf8"))
    tags_table = defaultdict(int)
    tags_num = 0
    for line in reader:
        for pair in pos_tag(list(map(wnl.lemmatize, line["tags"].split(" ")))):
            tags_table[pair[1]] += 1
            tags_num += 1
    for key in tags_table:
        tags_table[key] /= tags_num
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(tags_table)), tags_table.values(), align='center')
    plt.xticks(range(len(tags_table)), tags_table.keys())
    plt.xlabel("Parts of speech")
    plt.ylabel("Percent of words")
    plt.suptitle(csv_name)
    plt.savefig(join("charts", "part_of_speech", csv_name + ".png"))

for csv_name in ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']:
    print_tags(join("data", csv_name + ".csv"), csv_name)
