import numpy as np
from data import Corpus
START_CORPUS = 'trial_data_public/corpora/english/corpus1/corpus1.txt'
END_CORPUS = 'trial_data_public/corpora/english/corpus2/corpus2.txt'

def main():
    corpus1 = Corpus(START_CORPUS)
    corpus2 = Corpus(END_CORPUS)
    vocab = list((corpus1.word_counts + corpus2.word_counts).keys())
    

if '__name__' == '__main__':
    main()
