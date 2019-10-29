import nltk
from collections import Counter
from scipy.sparse import dok_matrix

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
#                cooccurences[b,a] += 1
    reciprocal = 1/sum(counts.values())
    independent_probabilities = {i:o * reciprocal for i, o in zip(counts.keys(),counts.values())}
    joint_probabilities = cooccurences.multiply(reciprocal)
    return cooccurences
