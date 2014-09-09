#!/usr/bin/env python
# http://www.nltk.org/
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#tokenizing-text-with-scikit-learn
# http://scikit-learn.org/stable/modules/cross_validation.html

import nltk.data
import sklearn, pprint, random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn import datasets

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# list of classes, file contents, and preamble delineators
fulltext = {"Conan Doyle":("pg1661.txt","ADVENTURE I"), 
            "Jane Austen":("pg31100.txt", "Chapter 1")};
sent = {};

# load books
for k,v in fulltext.items():
    # load the text and drop the preamble
    fulltext[k] = open(v[0]).read().split(v[1],1)[1];
    # split the text up into sentances
    fulltext[k] = tokenizer.tokenize(fulltext[k]);
    for c in fulltext[k]:
        sent[c] = k;


#pprint.pprint(sent);
count_vect = CountVectorizer(lowercase=True, stop_words="english",)

a  = count_vect.fit_transform(sent.keys())
#print count_vect.vocabulary_
print a.shape

Ntrain = 30000
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(a[0:Ntrain], sent.values()[0:Ntrain])
#clf = MultinomialNB().fit(a, sent.keys())

Ntest = 300
predicted = clf.predict(a[Ntrain:Ntrain+Ntest])
actual = sent.values()[Ntrain:Ntrain+Ntest];
rlist = zip(predicted,predicted == actual)

rstat = dict((i,rlist.count(i)) for i in rlist);
pprint.pprint( rstat )

tot = len(predicted)
ncorrect = sum( map( lambda x: rstat[x], filter( lambda x: x[1], rstat.keys() ) ) );
nwrong = tot - ncorrect;
print "correct classification: %f percent, incorrect: %f percent"%(100.0*ncorrect/tot, 100.0*nwrong/tot)

#X_train_counts = count_vect.fit_transform(d.values())
#print X_train_counts.size
#print count_vect.get_stop_words();
#print count_vect.get_feature_names();

#print dir(count_vect)
#print count_vect.stop_words
#print X_train_counts

#print d;


