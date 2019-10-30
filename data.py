import nltk
import numpy as np
from collections import Counter
from scipy.sparse import dok_matrix, coo_matrix, save_npz
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
    idx_to_word = list(word_to_idx.keys())
    print('Computing PPMI...')
    ppmi = dok_matrix(shape)
    for i, j in zip(np.nonzero(cooccurences)[0], np.nonzero(cooccurences)[1]):
            index_i = idx_to_word[i]
            index_j = idx_to_word[j]
            frequency = cooccurences[i,j]
            joint_probability = frequency / (frequency * vocab_size * vocab_size)
            probability_i = (frequency * counts[index_i]) / (frequency * vocab_size * vocab_size)
            probability_j = (frequency * counts[index_j]) / (frequency * vocab_size * vocab_size)
            denominator = probability_i * probability_j
            if denominator > 0:
                pmi = log(joint_probability / (probability_i * probability_j),2)
                ppmi[i,j] = max(0,pmi)
            else:
                ppmi[i,j] = 0
    print('finished computing')
    save_npz('cooccurence',coo_matrix(cooccurences))
    return cooccurences
