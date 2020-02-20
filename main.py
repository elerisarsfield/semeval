import nltk
import numpy as np
from corpus import Corpus, Word
from hdp import HDP

START_CORPUS = 'trial_data_public/corpora/english/corpus1/corpus1.txt'
END_CORPUS = 'trial_data_public/corpora/english/corpus2/corpus2.txt'
MAX_ITERS = 25


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
    for i in corpus.docs:
        i.init_partition(1)

    hdp = HDP(corpus.vocab_size, 'answer')
    hdp.init_partition(corpus.docs)
    for i in corpus.docs:
        i.topic_to_distribution(hdp.senses.shape[0])

    print('Done')
    it = 0
    stopping = 1.0
    print(f'Running Gibbs sampling for {MAX_ITERS} iterations...')
    while it < MAX_ITERS:
        it += 1
        for j in corpus.docs:
            for i in range(len(j.words)):
                hdp.sample_table(j, i, corpus.collocations[j.words[i]])
        if it % 10 == 0:
            corpus.save()
            print(f'Finished {it} iterations')
    for i in hdp.senses:
        i /= i.sum()
    print('Done')
    print('Generating scores for word senses...')
    words = dict()
    for j in corpus.docs:
        for i, p in enumerate(j.partition):
            origin = j.category
            sense = j.topic_to_global_idx[i]
            for w in p:
                if corpus.idx_to_word[w] in words:
                    if origin == 'reference':
                        words[corpus.idx_to_word[w]].senses[sense][0] += 1
                    else:
                        words[corpus.idx_to_word[w]].senses[sense][1] += 1
                else:
                    word = Word(corpus.idx_to_word[w], w, hdp.senses.shape[0])
                    if origin == 'reference':
                        word.senses[sense][0] += 1
                    else:
                        word.senses[sense][1] += 1
                    words[word.word] = word

    targets = ['walk', 'distance', 'small', 'god']
    for k, v in words.items():
        v = v.calculate()
        if k in targets:
            print(f'Score for {k}: {v[0]}')
    top_k = 50
    top = sorted(words, key=words.get, reverse=True)[:top_k]
    print(f'Top {top_k} most differing words:')
    print(top)


if __name__ == '__main__':
    main()
