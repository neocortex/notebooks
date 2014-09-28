""" Transformation classes for the Kaggle 'Sentiment Analysis on Movie
    Reviews' competition.

"""
from string import punctuation

from nltk.tokenize import word_tokenize
from nltk import wordnet
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import SGDClassifier
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
        self.clf = OneVsRestClassifier(SGDClassifier())

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
        according to the Harvard General Inquirer. """

    def __init__(self):
        """ Load The Harvard General Inquirer spreadsheet and save it in a
            DataFrame.

        """
        self.inquirer = pd.read_csv(
            'inqtabs.txt', delimiter='\t', index_col='Entry')
        self.inquirer.index = [i.lower() for i in self.inquirer.index]

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
        valence = []
        for phrase in phrases:
            words = phrase.split()
            pos = 0
            neg = 0
            for w in words:
                if w in self.inquirer.index:
                    if self.inquirer.ix[w]['Positiv'] == 'Positiv':
                        pos += 1
                    elif self.inquirer.ix[w]['Negativ'] == 'Negativ':
                        neg += 1
                    continue
                i = 0
                while True:
                    i += 1
                    word = w + '#{}'.format(i)
                    if word in self.inquirer.index:
                        if self.inquirer.ix[word]['Positiv'] == 'Positiv':
                            pos += 1
                        elif self.inquirer.ix[word]['Negativ'] == 'Negativ':
                            neg += 1
                    else:
                        break
            valence.append([pos, neg])
        return np.asarray(valence)
