from flask import Flask, Markup, render_template, request, jsonify

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.corpus import sentiwordnet
from nltk.corpus import wordnet
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

from twython import Twython
from twython import TwythonStreamer

import json

app = Flask(__name__)

consumer_key = 'Tbw54eQZKChqL1C6s0x3NriIv'
consumer_secret = 'aZzIKEq38cSHbA8uhfsEq8lifwm8MDpKby049AXdnCbjKDr1O2'
access_token = '856779498821373953-GKJYsGJNk6wEw6NaMy1KkPnBiqp9IUI'
access_secret = 'iyK9SIFRtbr1nStPEi7xZIW29Kdo4KWFrbWSSlVV5oVZl'

n_instances = 100

subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def training():
    sid = SentimentIntensityAnalyzer()
    sentence = request.form['sentence']
    saveFile = open('whisky.json','a')
    saveFile.write(sentence)
    saveFile.write('\n')
    saveFile.close()
    ss = sid.polarity_scores(sentence)
    return jsonify(ss)

@app.route('/papa', methods=['POST'])
def papa():
    return render_template('tango2.html')

@app.route('/november', methods=['GET','POST'])
def november():
    twitter = Twython(consumer_key, consumer_secret, access_token, access_secret)
    tango = request.form['tango']
    tweets = []
    prenegative = []
    prepositive = []
    negative = []
    neutral = []
    positive = []
    collection = []
    count = 0
    multiplier = 50

    results = twitter.cursor(twitter.search, q = tango, result_type = 'popular', lang = 'en')

    for result in results:
        if count == 25:
            break
        tweet = result['text']
        tweets.append(tweet)
        count+=1

    sid2 = SentimentIntensityAnalyzer()
    for tweet in tweets:
        ss2 = sid2.polarity_scores(tweet)
        for key, value in ss2.items():
            if key == 'neg':
                prenegative.append(value)
            if key == 'neu':
                neutral.append(value)
            if key == 'pos':
                prepositive.append(value)

    for x in prepositive:
        multiplier *= float(x)
        positive.append(multiplier)
        multiplier = 50

    for x in prenegative:
        multiplier *= float(x)
        negative.append(multiplier)
        multiplier = 50
        
    collection = tweets + positive + neutral + negative

    return jsonify(collection)

@app.route('/hotel', methods=['GET','POST'])
def roger():
    saveFile = open('whisky.json','r')
    lima = []
    lima = saveFile.read()
    saveFile.close()
    sid4 = SentimentIntensityAnalyzer()
    ss4 = sid4.polarity_scores(lima)
    return jsonify(ss4)

@app.route('/golf', methods=['GET','POST'])
def gama():
    return render_template('golf.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
