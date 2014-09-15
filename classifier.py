#!/usr/bin/env python
#
# Naive Bayes Author Classification from text
#
#  Copyright 2014 Tim O'Shea
# 
#  This classifier is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3, or (at your option)
#  any later version.
# 
#  This software is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
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

L = 20;
#Lr = range(0,L)
Lr = [9];
#L = 10;

#for ll in range(0,L):


for ll in Lr:
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

    # option one (force min_df = 2 to reduce features)
    #count_vect = CountVectorizer2(lowercase=True, stop_words="english", strip_accents="ascii", min_df=2)

    # option two (cap max_features at 100)
    #count_vect = CountVectorizer2(lowercase=True, stop_words="english", strip_accents="ascii", min_df=5, max_features=500)

    for F in range(1,100):
    #for F in range(100,10100,100):
#    for F in [100,200,300,400,500,600,700,800,900,1000,1500,2000]:
        print "F = %d"%(F)
    #count_vect = CountVectorizer2(lowercase=True, stop_words="english", strip_accents="ascii", min_df=5)
    #count_vect = CountVectorizer2(lowercase=True, stop_words="english", strip_accents="ascii", min_df=1)
        count_vect = CountVectorizer2(lowercase=True, stop_words="english", strip_accents="ascii", max_features=F)
        #count_vect = CountVectorizer2(lowercase=True, stop_words="english", strip_accents="ascii", min_df=1)
    #print count_vect
        a  = count_vect.fit_transform(sent.keys())
        print "Freq vect shape: " + str(a.shape)
        target =  numpy.array(sent.values())
    
    # set up clasifier
        from sklearn.naive_bayes import MultinomialNB
    
        avals = {0:"ML",
            0.5:"Good-Turing",
            1:"Laplace",
            2:"Two"};
        avals = {1:"Laplace"};
    
        for alpha,aname in avals.items():
        
            clf = MultinomialNB(alpha=alpha)
    
        # train and test classifier / cross validation
            print "Cross validating NB classifier... (alpha=%f [%s])"%(alpha,aname)
            scores = cross_validation.cross_val_score( clf, a, target, cv=10)
        #scores = cross_validation.cross_val_score( clf, a, target, cv=5)
            print scores
            print("Accuracy: %0.4f (+/- %0.4f)  [Mis-classification = %0.4f]" % (scores.mean(), scores.std() * 2, 1-scores.mean()))




