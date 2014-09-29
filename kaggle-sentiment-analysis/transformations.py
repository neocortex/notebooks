""" Transformation classes for the Kaggle 'Sentiment Analysis on Movie
    Reviews' competition.

    All classes inherit from ``sklearn.base.TransformerMixin`` and implement a
    `fit` and a `transform` function. The inheritance from `TransformerMixin`
    is not absolutely neccessary, but it brings the advantage of providing you
    with a `fit_transform` function for free.

"""
from cPickle import load
import csv
from string import punctuation

from nltk.tokenize import word_tokenize
from nltk import wordnet
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class PhraseCleaner(TransformerMixin):
    """ A stateless transform class. Clean raw movie review phrases by removing
        punctuation and converting to lowercase.

    """
    def fit(self, X, y=None):
        """ Stateless transform, no operation performed here. """
        return self

    def transform(self, phrases):
        """ Remove punctuation from each phrase and convert to lowercase.

            :param phrases: A list of strings corresponding to the review
                            phrases.
            :return: A list of 'cleaned' phrases.

        """
        cleaned_phrases = []
        for phrase in phrases:
            phrase = phrase.translate(None, punctuation)
            words = word_tokenize(phrase)
            cleaned_phrases.append(' '.join(words).lower())
        return cleaned_phrases


class ClassDistanceMapper(TransformerMixin):
    """ Fit a OneVsRestClassifier for each sentiment class (against all others
        combined) and return the distances from the decision boundary for each
        class. Hence, this transformation can be seen as a dimensionality
        reduction from #words to #sentiment_classes (=5).

    """

    def __init__(self):
        """ Initialize a one-vs-rest multiclass classifer with a
            SGDClassifier. The choice of the SGDclassifier here is arbitrary,
            any other classifier might work as well.

        """
        self.clf = OneVsRestClassifier(LogisticRegression())

    def fit(self, X, y):
        """ Fit the multiclass classifier. """
        self.clf.fit(X, y)
        return self

    def transform(self, X):
        """ Return the distance of each sample from the decision boundary for
            each class.

        """
        return self.clf.decision_function(X)


class SynsetsMapper(TransformerMixin):
    """ For each word in a phrase, return its WordNet synsets.  """

    def fit(self, X, y=None):
        """ Stateless transform: No operation performed here. """
        return self

    def transform(self, phrases):
        """ Return a string containing all synsets of each word in a phrase.

            :param phrases: A list of strings containing the phrases
                            of the moview reviews.
            :return: A list of strings containing all synsets of the phrases.

        """
        ssets = []
        for phrase in phrases:
            words = phrase.split()
            wss = []
            for w in words:
                ss = wordnet.wordnet.synsets(w)
                wss.extend([str(s.name().split('.')[0]) for s in ss])
            ssets.append(' '.join(wss))
        return ssets


class InquirerValence(TransformerMixin):
    """ Count the number of positive and negative words in each phrase
        according to the Harvard General Inquirer
        (http://www.wjh.harvard.edu/~inquirer/).

    """
    def __init__(self):
        """ Load the Harvard General Inquirer spreadsheet and save it in a
            DataFrame. Convert its index column (the words) to lowercase.

        """
        self.harvard_inquirer = pd.read_csv(
            'inqtabs.txt', delimiter='\t', index_col='Entry')
        self.harvard_inquirer.index = [
            i.lower() for i in self.harvard_inquirer.index]

    def fit(self, X, y=None):
        """ Stateless transform: No operation performed here. """
        return self

    def transform(self, phrases):
        """ For each word in a phrase, check if it is contained in the
            inquirer DataFrame and check if it has positive or negative
            valence (if any at all).

            :param phrases: A list of strings containing the phrases of the
                            moview reviews.
            :return: A numpy.ndarray containing number of positive and negative
                     words in each phrase.

        """
        harvard_valence = []
        for phrase in phrases:
            words = phrase.split()
            self.pos = self.neg = 0
            for w in words:
                if self._check_valence(w):
                    continue
                i = 0
                while True:
                    i += 1
                    word = w + '#{}'.format(i)
                    if not self._check_valence(word):
                        break
            harvard_valence.append([self.pos, self.neg])
        return np.asarray(harvard_valence)

    def _check_valence(self, word):
        if word in self.harvard_inquirer.index:
            if self.harvard_inquirer.ix[word]['Positiv'] == 'Positiv':
                self.pos += 1
            elif self.harvard_inquirer.ix[word]['Negativ'] == 'Negativ':
                self.neg += 1
            return True
        return False


class LiuOpinion(TransformerMixin):
    """ Count the number of positive and negative words in each phrase
        according to the Bing Liu's sentiment lexicon
        (http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html).

    """
    def __init__(self):
        self.pos_list = []
        self.neg_list = []

        with open('positive-words.txt') as f:
            reader = csv.reader(row for row in f if not row.startswith(';'))
            for row in reader:
                if row:
                    self.pos_list.append(row[0])
        with open('negative-words.txt') as f:
            reader = csv.reader(row for row in f if not row.startswith(';'))
            for row in reader:
                if row:
                    self.neg_list.append(row[0])

    def fit(self, X, y=None):
        """ Stateless transform: No operation performed here. """
        return self

    def transform(self, phrases):
        """ For each word in a phrase, check if it is contained in either the
            positive or negative list of Bing Liu's sentiment lexicon.

            :param phrases: A list of strings containing the phrases of the
                            moview reviews.
            :return: A numpy.ndarray containing number of positive and negative
                     words in each phrase.

        """
        opinion = []
        for phrase in phrases:
            words = phrase.split()
            pos = neg = 0
            for w in words:
                if w in self.pos_list:
                    pos += 1
                elif w in self.neg_list:
                    neg += 1
            opinion.append([pos, neg])
        return np.asarray(opinion)


class SentiWordNetMapper(TransformerMixin):

    def __init__(self):
        self.swn_dict = load(file('sentiwordnet_cleaned.pkl'))

    def fit(self, X, y=None):
        """ Stateless transform: No operation performed here. """
        return self

    def transform(self, phrases):
        """ For each word in a phrase, check if it is contained in the
            inquirer DataFrame and check if it has positive or negative
            valence (if any at all).

            :param phrases: A list of strings containing the phrases of the
                            moview reviews.
            :return: A numpy.ndarray containing number of positive and negative
                     words in each phrase.

        """
        opinion = []
        for phrase in phrases:
            words = phrase.split()
            pos = neg = 0
            for w in words:
                if w in self.swn_dict.keys():
                    pos += self.swn_dict[w][0]
                    neg += self.swn_dict[w][1]
            opinion.append([pos, neg])
        return np.asarray(opinion)
