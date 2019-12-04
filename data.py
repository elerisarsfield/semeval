import math
import random
import nltk
import numpy as np
from collections import Counter
from functools import reduce
from scipy.sparse import dok_matrix
from scipy.spatial.distance import cosine
from scipy.stats import dirichlet
#from processes import CRF


class Corpus:
    def __init__(self, source, stopword_k=100,alpha=1):
        """Basic preprocessing and identify collocations"""
        self.stopword_k = stopword_k
        self.alpha = alpha
        corpus_data = self.split_words(source)
        self.collocations(corpus_data)
        for i, w in enumerate(self.idx_to_word):
            self.senses[i] = self.get_clusters(w)

    def split_words(self, filepath):
        """Split text into sentences and extract word counts"""
        with open(filepath, 'r') as f:
            sentences = [i.strip() for i in f]
            words = [j for i in sentences for j in nltk.word_tokenize(i)]
            self.word_counts = Counter(words)
            return sentences

    def collocations(self, corpus, window_size=10):
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
                pmi = math.log(joint_probability / (probability_i * probability_j),2)
                ppmi[i,j] = max(0,pmi)
            else:
                ppmi[i,j] = 0
        print('finished computing')
        self.collocations = cooccurences
        self.shape = cooccurences.shape
        self.senses = [None] * self.shape[0]
        
    def get_clusters(self,word):
        idx = self.word_to_idx[word]
        observations = self.collocations[idx]
        senses = []
        for n,i in enumerate(np.nonzero(observations)[1]):
            new_sense_p = self.alpha/(n+self.alpha)
            if new_sense_p == 1.0:
                senses.append([i])
                continue
            similarities = [0] * len(senses)
            for loc,j in enumerate(senses):
                sense_size = len(j)
                word_p = np.sum(self.collocations[i])
                sense_p = sense_size/ (n + self.alpha)
#                sense_similarity = sum(map(lambda x: (np.dot(np.reshape(self.collocations[i].toarray(), -1), np.reshape(self.collocations[x].toarray(), -1))) / (self.collocations[i].size * self.collocations[x].size), j)) + self.alpha
                sense_similarity = sum(map(lambda x: self.collocations[x,i] * self.word_counts[self.idx_to_word[i]], j))

                similarities[loc] = (sense_similarity * sense_size + self.alpha) / (self.alpha + n)
            similarities.append(new_sense_p)
            # print(similarities)
            prior = dirichlet(similarities).rvs()
            assert math.isclose(np.sum(prior), 1)
            prior = np.reshape(prior, -1)
            assignment = random.random()
            if assignment > np.sum(prior[:-1]):
                senses.append([i])
            else:
                assert len(prior) == len(senses) + 1
                curr = 0
                for j,p in enumerate(prior[:-1]):
                    curr += p
                    if curr > assignment:
                        senses[j].append(i)
                        break
                else:
                    print('Error in cluster assignment')

                
#        print([list(map(lambda x: self.idx_to_word[x], i)) for i in senses])
        return senses
