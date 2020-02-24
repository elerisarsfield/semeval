import nltk
import argparse
from corpus import Corpus, Word
from hdp import HDP

parser = argparse.ArgumentParser()
parser.add_argument(
    'start_corpus', type=str, help='address of the older (reference) corpus')
parser.add_argument('end_corpus', type=str,
                    help='address of the newer (focus) corpus')
parser.add_argument('targets', type=str, help='address of the target words')
parser.add_argument('output', type=str, help='address to write output to')
parser.add_argument('--max_iters', type=int, metavar='N', default=25,
                    help='maximum number of iterations to run sampling for')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='alpha value, default 1.0')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='gamma value, default 1.0')
parser.add_argument('--eta', type=float, default=0.1,
                    help='eta value, default 0.1')
parser.add_argument('--window_size', metavar='W', type=int, default=10,
                    help='size of context window to use, default 10')
parser.add_argument('--floor', type=int, metavar='F', default=1,
                    help='minimum number of occurrences to be considered, default 1')
args = parser.parse_args()


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
    corpus = Corpus(args.start_corpus, args.end_corpus, args.output)
    print('Setting up initial partition...')
    for i in corpus.docs:
        i.init_partition(args.alpha)

    hdp = HDP(corpus.vocab_size, args.output)
    hdp.init_partition(corpus.docs)
    for i in corpus.docs:
        i.topic_to_distribution(hdp.senses.shape[0])

    print('Done')
    it = 0
    stopping = 1.0
    print(f'Running Gibbs sampling for {args.max_iters} iterations...')
    while it < args.max_iters:
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
