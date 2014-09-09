#!/usr/bin/env python
# http://www.nltk.org/
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#tokenizing-text-with-scikit-learn
# http://scikit-learn.org/stable/modules/cross_validation.html

import nltk.data
import sklearn, pprint, random, numpy, scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn import datasets

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stem = nltk.stem.PorterStemmer()

class CountVectorizer2(CountVectorizer):
    def build_analyzer(self):
        analyzer = CountVectorizer.build_analyzer(self)
        return lambda x: (stem.stem(y) for y in analyzer(x))


L = 10;

for ll in range(0,L):
    l = ll + 1;

    print "Showing results for classification using L=%d sentence examples"%(l)

    # list of classes, file contents, and preamble delineators
    fulltext = {"Conan Doyle":("pg1661.txt","ADVENTURE I"), 
                "Jane Austen":("pg31100.txt", "Chapter 1")};
    fulltextj = {};
    sent = {};

    # load books
    for k,v in fulltext.items():
        # load the text and drop the preamble
        fulltext[k] = open(v[0]).read().split(v[1],1)[1];
        # split the text up into sentances
        fulltext[k] = tokenizer.tokenize(fulltext[k]);
        fulltextj[k] = [];

        # join l sentances to make one test case
        for i in range(0,len(fulltext[k])/l):
            fulltextj[k].append( reduce( lambda x,y: x+y, fulltext[k][i*l:(i+1)*l] ) )
                
        for c in fulltextj[k]:
            sent[c] = k;

    # convert to indexed form
    count_vect = CountVectorizer2(lowercase=True, stop_words="english", strip_accents="ascii")
    a  = count_vect.fit_transform(sent.keys())
    target =  numpy.array(sent.values())
    
    # set up clasifier
    from sklearn.naive_bayes import MultinomialNB
    
    avals = {0:"ML",
            0.5:"Good-Turing",
            1:"Laplace",
            2:"Two"};
    
    for alpha,aname in avals.items():
        
        clf = MultinomialNB(alpha=alpha)
    
        # train and test classifier / cross validation
        print "Cross validating NB classifier... (alpha=%f [%s])"%(alpha,aname)
        scores = cross_validation.cross_val_score( clf, a, target, cv=10)
        #scores = cross_validation.cross_val_score( clf, a, target, cv=5)
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)  [Mis-classification = %0.2f]" % (scores.mean(), scores.std() * 2, 1-scores.mean()))




