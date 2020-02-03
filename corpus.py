import math
import random
import nltk
import numpy as np
import collections
import os
import pickle
from scipy import stats, sparse


class Corpus:
    def __init__(self, reference, focus, output, floor=1,
                 window_size=10):
        """Basic preprocessing and identify collocations"""
        self.floor = floor + 1
        self.sentences = self.get_documents(
            reference) + self.get_documents(focus)
        self.total_words = 0
        self.preprocess()
        self.vocab_size = len(self.word_counts)
#        self.collocations(sentences)
#        self.base = np.fromiter((np.sum(self.collocations[x])
#                                 for x in range(self.shape[0])), float)
        # self.base = dirichlet(self.base,np.random.gamma(
        # self.base.shape,self.gamma)).rvs()
        self.base = stats.dirichlet([1] * self.vocab_size)
        self.senses = [sparse.dok_matrix(
            (self.vocab_size, 1))] * self.vocab_size
 #       for m, s in enumerate(sentences):
  #          w = nltk.word_tokenize(s)
#            print(f'Sentence: {s}')
#            print(f'Words: {w}')
   # for i, _ in enumerate(w):
     #           window_start = max(i-(window_size//2), 0)
      #          window_end = min(len(w) - 1, i + (window_size // 2))
       #         context = w[window_start:window_end]
 #           if m > 10:
  #              break
        # for i, w in enumerate(self.idx_to_word):
        # self.senses[i] = self.get_clusters(w)

    def get_documents(self, filepath):
        """Split text into sentences and extract word counts"""
        with open(filepath, 'r') as f:
            sentences = [i.strip() for i in f]
            return sentences

    def preprocess(self):
        words = [j for i in self.sentences for j in nltk.word_tokenize(i)]
        self.word_counts = collections.Counter(words)
        stopwords = set(nltk.corpus.stopwords.words('english'))
        for i in self.word_counts.most_common()[::-1]:
            if i[1] < self.floor:
                stopwords.add(i[0])
        for i, s in enumerate(self.sentences):
            new = [i for i in s.split(' ') if i not in stopwords]
            self.sentences[i] = ' '.join(new).strip()
            self.total_words += len(new)
        self.word_counts = collections.Counter(
            [j for i in self.sentences for j in nltk.word_tokenize(i)])

    def get_clusters(self, word):
        idx = self.word_to_idx[word]
        observations = self.collocations[idx]
        senses = []
        for n, i in enumerate(np.nonzero(observations)[1]):
            new_sense_p = self.alpha/(n+self.alpha)
            if new_sense_p == 1.0:
                senses.append([i])
                continue
            similarities = [0] * len(senses)
            for loc, j in enumerate(senses):
                sense_size = len(j)
                word_p = np.sum(self.collocations[i])
                sense_p = sense_size / (n + self.alpha)
                # sense_similarity = sum(map(lambda x: (
                # np.dot(np.reshape(self.collocations[i].toarray(), -1),
                # np.reshape(self.collocations[x].toarray(), -1))) /
                # (self.collocations[i].size * self.collocations[x].size),
                # j)) + self.alpha
                sense_similarity = sum(
                    map(lambda x: self.collocations[x, i] *
                        self.word_counts[self.idx_to_word[i]], j))

                similarities[loc] = (
                    sense_similarity * sense_size + self.alpha
                ) / (self.alpha + n)
            similarities.append(new_sense_p)
            # print(similarities)
            prior = stats.dirichlet(similarities, self.alpha).rvs()
            assert math.isclose(np.sum(prior), 1)
            prior = np.reshape(prior, -1)
            assignment = random.random()
            if assignment > np.sum(prior[:-1]):
                senses.append([i])
            else:
                assert len(prior) == len(senses) + 1
                curr = 0
                for j, p in enumerate(prior[:-1]):
                    curr += p
                    if curr > assignment:
                        senses[j].append(i)
                        break
                else:
                    print('Error in cluster assignment')


#        print([list(map(lambda x: self.idx_to_word[x], i)) for i in senses])
        return senses

    def save(self):
        out = os.path.join(self.output, 'model.pkl')
        pickle.dump(self, open(out, 'wb'))
