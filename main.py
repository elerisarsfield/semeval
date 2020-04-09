"""Runs the project"""
import nltk
import scipy.spatial.distance as dist
import numpy as np
import argparse
import time
import os
import utils
from corpus import Corpus, Word
from hdp import HDP

# command line options
parser = argparse.ArgumentParser()
parser.add_argument(
    'start_corpus', type=str, help='address of the older (reference) corpus')
parser.add_argument('end_corpus', type=str,
                    help='address of the newer (focus) corpus')
parser.add_argument('--semeval_mode', type=bool,
                    help='True if the project is being used for SemEval 2020 Task 1, False if the project is being used for general inference, default False', default=False, metavar='M')
parser.add_argument('targets', type=str,
                    help='address of the target words', nargs='?')
parser.add_argument('output', type=str, help='address to write output to')
parser.add_argument('--top_k', type=int, metavar='k',
                    default=25, help='number of words to output')
parser.add_argument('--max_iters', type=int, metavar='N', default=25,
                    help='maximum number of iterations to run sampling for')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='alpha value, default 1.0')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='gamma value, default 1.0')
parser.add_argument('--window_size', metavar='W', type=int, default=10,
                    help='size of context window to use, default 10')
parser.add_argument('--floor', type=int, metavar='F', default=1,
                    help='minimum number of occurrences to be considered, default 1')
parser.add_argument('--threshold', type=float, metavar='T', default=0.6,
                    help='minimum score for  before a word is considered to have a novel sense')

args = parser.parse_args()
if args.semeval_mode and 'targets' not in vars(args):
    parser.error('targets arg is required when in SemEval mode')


def main():
    """Run the project"""
    start_time = time.time()

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    save_path = os.path.join(args.output, 'saves')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(args.output, 'task1')):
        os.makedirs(os.path.join(args.output, 'task1'))
    if not os.path.exists(os.path.join(args.output, 'task2')):
        os.makedirs(os.path.join(args.output, 'task2'))

    print('Loading words...')
    corpus = Corpus(args.start_corpus, save_path,
                    args.end_corpus, args.floor, args.window_size)
    print('Setting up initial partition...')
    for i in corpus.docs:
        i.init_partition(args.alpha)

    hdp = HDP(corpus.vocab_size, save_path,
              alpha=args.alpha, gamma=args.gamma)
    hdp.init_partition(corpus.docs)
    print('Done')
    it = 0
    print(f'Running Gibbs sampling for {args.max_iters} iterations...')
    while it < args.max_iters:
        for j in corpus.docs:
            for i in range(len(j.words)):
                hdp.sample_table(j, i, corpus.collocations[j.words[i]])
        it += 1
        corpus.save()
        print(f'Iteration {it}/{args.max_iters}')
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
    print('Done.')
    if args.semeval_mode:
        targets = utils.get_targets(args.targets)
        results = []
        for i in range(len(targets)):
            t = targets[i][0]
            pos = targets[i][1]
            recombine = t+'_'+pos
            word = words[recombine]
            scores = word.senses[~np.all(word.senses == 0, axis=1)]

            dist_1 = scores[:, 0]
            dist_2 = scores[:, 1]
            jensenshannon = dist.jensenshannon(
                dist_1, dist_2)
            results.append((recombine, jensenshannon))

        with open(os.path.join(os.path.join(args.output, 'task1'),
                               'english.txt'), 'w') as f:
            for i in results:
                recombine = i[0]
                score = i[1]
                different = 1 if score > args.threshold else 0
                f.write(f'{recombine} {different}\n')

        with open(os.path.join(os.path.join(args.output, 'task2'), 'english.txt'), 'w') as f:
            for i in results:
                recombine = i[0]
                jensenshannon = i[1]
                f.write(f'{recombine} {jensenshannon:.4f}\n')

    else:
        for k, v in words.items():
            words[k] = v.calculate()
        top = sorted(words, key=words.get, reverse=True)[:args.top_k]
        with open(os.path.join(args.output, 'out.txt'), 'w') as f:
            f.write(f'Top {args.top_k} most differing words:')
            f.write('\n'.join(top))
    end_time = time.time()
    print(f'Ran project in {end_time - start_time} seconds')


if __name__ == '__main__':
    main()
