import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy
import logging

from nltk.corpus import stopwords # imports various modules for string cleaning
import re
import nltk.data

from bs4 import BeautifulSoup # removes stopwords
from gensim.models import Word2Vec

from gensim import models

# nltk.download()

# Read data from files
train = pd.read_csv( "labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0,
                              delimiter="\t", quoting=3 )

print(train["review"][0])

print("Read %d labeled train reviews, %d labeled test reviews, " \
"and %d unlabled reviews \n" %
(train["review"].size,
 test["review"].size,
 unlabeled_train["review"].size))

# Function to convert a document into a sequence of words
# Optionally removing stop words
# Returns a list of words
def review_to_wordlist(review, remove_stopwords=False ):
    # Removes HTML
    review_text = BeautifulSoup(review).get_text()
    # Removes non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # Converts words to lower case and splits them by whitecase to an array
    words = review_text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # Returns the list of words
    return(words)

# Loads the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle');

# Defines a function to split a review into parsed sentences
# splits a reiew into parsed sentences. Returns a list of sentences, where each
# sentence is a list of words
def review_to_sentences(review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.strip().decode('utf-8'))
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, \
             remove_stopwords ))
    return sentences

sentences = [] # initialize an empty list of sentences for the review_to_sentences()

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print(len(sentences))

# Training Model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s' , \
	level = logging.INFO);

num_features = 360 # word vector dimesionality
min_word_count = 40 # minimum word count
num_workers = 4 # number of threads to run in parallel
context = 10 # context window size
downsampling = 1e-3 # downsample setting for frequent words

print("Training model...");

model = Word2Vec(sentences, workers = num_workers, \
	size = num_features, min_count = min_word_count, \
	window = context, sample = downsampling)

model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)

# Word2Vec.load("testData.tsv")

print(model.doesnt_match("man woman child kitchen".split()))
print(model.most_similar("man"))
print(model.most_similar("computer"))



