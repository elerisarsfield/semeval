import nltk
import numpy as np
from collections import Counter, namedtuple
from scipy.sparse import dok_matrix
from scipy.spatial.distance import cosine
from math import log

SenseLocation = namedtuple("SenseLocation","idx probability")

class Corpus:
    def __init__(self, source, stopword_k=100,alpha=-0.1,theta=1):
        """Basic preprocessing and identify collocations"""
        self.stopword_k = stopword_k
        self.alpha = alpha
        self.theta = theta
        corpus_data = self.split_words(source)
        self.collocations(corpus_data)

    def split_words(self, filepath):
        """Split text into sentences and extract word counts"""
        with open(filepath, 'r') as f:
            sentences = [i.strip() for i in f]
            words = [j for i in sentences for j in nltk.word_tokenize(i)]
            self.word_counts = Counter(words)
            return sentences

    def collocations(self, corpus, window_size=4):
        """Build the co-occurence matrix"""
        vocab_size = len(self.word_counts)
        self.word_to_idx = {o:i for i,o in enumerate(self.word_counts.keys())}
        shape = (vocab_size, vocab_size)
        print('Starting co-occurence matrix build...')
        cooccurences = dok_matrix(shape)
        stopwords = [i[0] for i in self.word_counts.most_common(self.stopword_k)]
        self.word_counts = self.word_counts - Counter(self.word_counts.most_common(self.stopword_k))
        for i in corpus:
            tokens = [i for i in nltk.word_tokenize(i)]
            for j, k in enumerate(tokens):
                if k in stopwords:
                    continue
                window_start = max(0, j - (window_size // 2))
                window_end = min(len(tokens) - 1, j + (window_size // 2))
                occurences = tokens[window_start:window_end]
                for l in occurences:
                    if l in stopwords:
                        continue
                    if l != k:
                        a = self.word_to_idx[l]
                        b = self.word_to_idx[k]
                        cooccurences[a,b] += 1

        reciprocal = 1/sum(self.word_counts.values())
            
        self.idx_to_word = list(self.word_to_idx.keys())
        print('Computing PPMI...')
        ppmi = dok_matrix(shape)
        total = np.sum(cooccurences)
        for i, j in zip(np.nonzero(cooccurences)[0], np.nonzero(cooccurences)[1]):
            index_i = self.idx_to_word[i]
            index_j = self.idx_to_word[j]
            frequency = cooccurences[i,j]
            joint_probability = frequency / total
            probability_i = (frequency * self.word_counts[index_i]) / total
            probability_j = (frequency * self.word_counts[index_j]) / total
            denominator = probability_i * probability_j
            if denominator > 0:
                pmi = log(joint_probability / (probability_i * probability_j),2)
                ppmi[i,j] = max(0,pmi)
            else:
                ppmi[i,j] = 0
        print('finished computing')
        self.collocations = cooccurences
        self.shape = cooccurences.shape
        
    def get_clusters(self,word):
        idx = self.word_to_idx[word]
        observations = self.collocations[idx]
        senses = []
        for n,i in enumerate(np.nonzero(observations)[1]):
            new_sense_p = (self.theta + len(senses)*self.alpha)/(n+self.theta)
            best_sense = SenseLocation(0,0)
            for loc,j in enumerate(senses):
                sense_size = len(j)
                sense_p = (sense_size - self.alpha)/ (n + self.theta)
                sense_similarity = sum(map(lambda x: cosine(
                    self.collocations[i].toarray(),self.collocations[x].toarray()), j))
                sense_p *= sense_similarity
                if sense_p > best_sense.probability:
                    best_sense = SenseLocation(loc, sense_p)
            if new_sense_p > best_sense.probability:
                senses.append([i])
            else:
                senses[best_sense.idx].append(i)
        print([list(map(lambda x: self.idx_to_word[x], i)) for i in senses])

    def process(self):
        pass
