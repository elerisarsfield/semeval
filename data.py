from collections import Counter

def split_words(filepath):
    """Split text into sentences and extract word counts"""
    with open(filepath, 'r') as f:
        sentences = [i.strip() for i in f]
        words = [j for i in sentences for j in i.split(' ')]
        word_counts = Counter(words)
        return sentences, word_counts

def collocations(corpus, counts):
    word_to_idx = {o:i for i,o in enumerate(counts.keys())}
