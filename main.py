import nltk
from corpus import Corpus
from hdp import HDP

START_CORPUS = 'trial_data_public/corpora/english/corpus1/corpus1.txt'
END_CORPUS = 'trial_data_public/corpora/english/corpus2/corpus2.txt'
MAX_ITERS = 1000


def main():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    print('Loading words...')
    corpus = Corpus(START_CORPUS, END_CORPUS, 'answer')
    print('Setting up initial partition...')
    hdp = HDP(corpus.vocab_size)
    hdp.init_partition(corpus.sentences)
    hdp.gibbs(False, False)
    print('Done')
    it = 0
    print(f'Running Gibbs sampling for {MAX_ITERS} iterations')
    while it < MAX_ITERS:
        it += 1
        hdp.gibbs()
        likelihood = hdp.llikelihood()
    print('Done')


if __name__ == '__main__':
    main()
