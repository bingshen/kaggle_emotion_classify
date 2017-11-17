from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
from gensim.models import Word2Vec
import os
import nltk

if __name__ == '__main__':
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    unlabeled_df=pd.read_csv("data\\unlabeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    vocab_file=open("data/vocab.txt","w")
    vocab_file.write("<unk>\n<sos>\n<eos>\n")
    unique_word={}
    for review in labeled_df['review']:
        wordlist=Word2VecUtil.review_to_wordlist(review)
        for word in wordlist:
            if word not in unique_word:
                unique_word[word]=unique_word.get(word,0)+1
    for review in unlabeled_df['review']:
        wordlist=Word2VecUtil.review_to_wordlist(review)
        for word in wordlist:
            if word not in unique_word:
                unique_word[word]=unique_word.get(word,0)+1
    for review in test_df['review']:
        wordlist=Word2VecUtil.review_to_wordlist(review)
        for word in wordlist:
            if word not in unique_word:
                unique_word[word]=unique_word.get(word,0)+1
    for word in unique_word.keys():
        line=word+'\n'
        vocab_file.write(line)