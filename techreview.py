import random
import time

import nltk
import nltk.classify as cl
from nltk.corpus import (movie_reviews, pros_cons, sentence_polarity,
                         twitter_samples)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def twitter_data():
    return [(tweet, "positive") for tweet in twitter_samples.strings("positive_tweets.json")] + [
        (tweet, "negative") for tweet in twitter_samples.strings("negative_tweets.json")
    ]


def pc_data():
    return [(" ".join(sent), "positive") for sent in pros_cons.sents("IntegratedPros.txt")] + [
        (" ".join(sent), "negative") for sent in pros_cons.sents("IntegratedCons.txt")
    ]


def sp_data():
    return [(" ".join(sent), "positive") for sent in sentence_polarity.sents("rt-polarity.pos")] + [
        (" ".join(sent), "negative") for sent in sentence_polarity.sents("rt-polarity.neg")
    ]


def model_accuracy(classifier, gold):
    results = classifier.classify_many([fs for (fs, l) in gold])
    correct = [l == r for ((fs, l), r) in zip(gold, results)]
    if correct:
        return sum(correct) / len(correct)
    else:
        return 0


classifiers = [
    (cl.NaiveBayesClassifier, "Naive Bayes"),
    (cl.DecisionTreeClassifier, "Decision Tree"),
    (cl.MaxentClassifier, "Max Ent")
]

data = [
    twitter_data,
    pc_data,
    sp_data
]


def analyze():
    sia = SentimentIntensityAnalyzer()
    for ds in data:
        print(ds)
        dataset = ds()
        random.seed(123456)
        random.shuffle(dataset)
        train_index = int(len(dataset) * .9)
        train = dataset[:train_index]
        test = dataset[train_index:]

        pol_scores = [(sia.polarity_scores(n), g) for (n, g) in train]
        s = sum(item[0]['pos'] for item in pol_scores)/train_index
        print("SIA Accuracy: ", s)       
        for classifier in classifiers:
            print(classifier)
            train_start = time.time()
            obj = classifier[0].train(pol_scores)
            train_end = time.time()
            print(classifier[1] + " Training Time: ", train_end - train_start)
            test_start = time.time()
            acc = model_accuracy(obj, [(sia.polarity_scores(n), g) for (n, g) in test])
            test_end = time.time()
            print(classifier[1] + " Testing Time: ", test_end - test_start)
            print(classifier[1] + " Accuracy: ", acc)

for i in range(5):
    analyze()
