__author__ = 'Manos'

import os
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import pickle
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd

#path of the txt files
path=r"/Users/Manos/projects/data_mining/books"

stops = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
stops.update(['"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

#remove non asci caracters
def removeNonAscii(document):
    "".join(i for i in document if ord(i)<128)
    wordBag = re.sub('[^A-Za-z0-9]+', ' ', document)
    return wordBag

#replacing double spaces from txts with one space
def removeDoubleSpaces(document):
    wordBag=' '.join(document.split())
    return wordBag

#replacing capital letters with lowers
def lowerCase(document):
    wordBag= document.lower()
    return wordBag

#tokenize the document
#create list of words and compare them with stopwords
def removeStopWords(document):

    filtered_words = nltk.word_tokenize(document)
    # result = ' '.join(filtered_words)

    afterstems = [stemmer.stem(t) for t in filtered_words if len(t)>2]
    final_result = ' '.join(afterstems)
    return final_result
#     # return listOfWords

#find every .txt file in the folder
for files in os.listdir(path):
    if files.endswith(".txt"):
        print(files)

list_all = list()
for files in os.listdir(path):
    if files.endswith(".txt"):
        with open(files,'r+')as files:
            document = files.read()

            documentAfterAscii = removeNonAscii(document)
            documentAfterSpaces = removeDoubleSpaces(documentAfterAscii)
            synopsis_pre = lowerCase(documentAfterSpaces)
            synopsis = removeStopWords(synopsis_pre)
           

            list_all.append(synopsis)

        pickle.dump(list_all, open( "synopsis.p" , "wb" ) )






