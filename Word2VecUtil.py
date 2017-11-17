from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import re

class Word2VecUtil(object):
    stops=set(stopwords.words("english"))
    negators=[line.strip() for line in open("data\\negator.txt")]
    tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    @staticmethod
    def review_to_wordlist(review):
        review_text=BeautifulSoup(review,"lxml").get_text()
        review_text=re.sub('n\'t',' not',review_text)
        review_text=re.sub('[^a-zA-z]',' ',review_text)
        words=review_text.lower().split()
        return words
    @staticmethod
    def review_to_sentences(review):
        raw_sentences=Word2VecUtil.tokenizer.tokenize(review.strip())
        sentences=[]
        for sentence in raw_sentences:
            if len(sentence)>0:
                sentences.append(Word2VecUtil.review_to_wordlist(sentence))
        return sentences