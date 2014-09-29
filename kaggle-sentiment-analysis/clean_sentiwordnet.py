from cPickle import dump
import pandas as pd

df = pd.read_csv('SentiWordNet_3.0.0_20130122.txt', delimiter='\t',
                 index_col='SynsetTerms')

sentiwordnet_dict = dict()
for i, ix in enumerate(df.index):
    print 'Word {} / {} ...'.format(i+1, len(df.index))
    entry = df.ix[ix]
    try:
        if (entry['PosScore'] != 0) or (entry['NegScore'] != 0):
            words = ix.split(' ')
            for w in words:
                w = w.split('#')[0]
                if w.isalpha():
                    sentiwordnet_dict[w] = (
                        entry['PosScore'], entry['NegScore'])
    except ValueError:
        if (entry['PosScore'].max() != 0) or (entry['NegScore'].max() != 0):
            words = ix.split(' ')
            for w in words:
                w = w.split('#')[0]
                if w.isalpha():
                    sentiwordnet_dict[w] = (
                        entry['PosScore'].max(), entry['NegScore'].max())

dump(sentiwordnet_dict, file('sentiwordnet_cleaned.pkl', 'wb'))
