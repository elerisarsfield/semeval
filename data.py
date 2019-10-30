import nltk
from collections import Counter
from scipy.sparse import dok_matrix
from math import log

def split_words(filepath):
    """Split text into sentences and extract word counts"""
    with open(filepath, 'r') as f:
        sentences = [i.strip() for i in f]
        words = [j for i in sentences for j in nltk.word_tokenize(i)]
        word_counts = Counter(words)
        return sentences, word_counts

def collocations(corpus, counts, window_size=4):
    """Build the co-occurence matrix"""
    vocab_size = len(counts)
    word_to_idx = {o:i for i,o in enumerate(counts.keys())}
    shape = (vocab_size, vocab_size)
    print('Starting co-occurence matrix build...')
    cooccurences = dok_matrix(shape)
    for i in corpus:
        tokens = nltk.word_tokenize(i)
        for j, k in enumerate(tokens):
            window_start = max(0, j - (window_size // 2))
            window_end = min(len(tokens) - 1, j + (window_size // 2))
            occurences = tokens[window_start:window_end]
            for l in occurences:
                if l != k:
                    a = word_to_idx[l]
                    b = word_to_idx[k]
                    cooccurences[a,b] += 1
                    cooccurences[b,a] += 1
    reciprocal = 1/sum(counts.values())
    independent_probabilities = {i:o * reciprocal for i, o in zip(counts.keys(),counts.values())}
    joint_probabilities = cooccurences.multiply(reciprocal)
    pmi = lambda x,y: log(joint_probabilities[word_to_idx[x],word_to_idx[y]]/(independent_probabilities[x]*independent_probabilities[y]),2)
    print('Computing PPMI...')
    for i in counts.keys():
        for j in counts.keys():
            index_i = word_to_idx[i]
            index_j = word_to_idx[j]
            if joint_probabilities[index_i,index_j] != 0:
                cooccurences[index_i,index_j] = pmi(i,j)
    print('finished computing')
    return cooccurences
