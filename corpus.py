import math
import random
import nltk
import numpy as np
import collections
import os
import pickle
from scipy import stats, sparse


class Document():
    def __init__(self, idx, doc):
        self.idx = idx
        self.partition = []
        self.topic_to_global_idx = []
        self.global_to_local_topic = dict()
        counts = collections.Counter(doc)
        self.words = list(counts.keys())
        self.counts = list(counts.values())
        self.length = len(self.words) - 1
        self.total = sum(self.counts)
        self.topics = []

    def init_partition(self, alpha):
        N = 0
        for i in self.words:
            N += 1
            prior = [0] * len(self.partition)
            for j in range(len(self.partition)):
                probability = len(self.partition[j])/(N+alpha-1)
                prior[j] = probability
            new = alpha/(N+alpha-1)
            prior.append(new)
            table = random.random()
            if table > sum(prior[:-1]):
                self.partition.append([i])
            else:
                curr = 0
                for j, p in enumerate(prior):
                    curr += p
                    if curr > table:
                        self.partition[j].append(i)
                        break

    def topic_to_distribution(self, topics, eta=0.1):
        self.topics = np.fromiter((eta*self.topic_to_global_idx.count(
            i) if i in self.topic_to_global_idx else 0 for i in np.arange(topics)),
            dtype=np.float32, count=topics)


class Corpus:
    def __init__(self, reference, focus, output, floor=1,
                 window_size=10):
        """Basic preprocessing and identify collocations"""
        self.floor = floor
        self.total_words = 0
        self.vocab_size = 0
        self.word_counts = None
        self.docs = []
        self.word_to_idx = None
        self.sentences = self.get_documents(
            reference) + self.get_documents(focus)
        self.collocations(self.sentences)
#        self.base = np.fromiter((np.sum(self.collocations[x])
#                                 for x in range(self.shape[0])), float)
        # self.base = dirichlet(self.base,np.random.gamma(
        # self.base.shape,self.gamma)).rvs()
#        self.base = stats.dirichlet([1] * self.vocab_size)
#        self.senses = [sparse.dok_matrix(
 #           (self.vocab_size, 1))] * self.vocab_size
   # for i, _ in enumerate(w):
     #           window_start = max(i-(window_size//2), 0)
      #          window_end = min(len(w) - 1, i + (window_size // 2))
       #         context = w[window_start:window_end]
 #           if m > 10:
  #              break

    def get_documents(self, filepath):
        """Split text into sentences and extract word counts"""
        with open(filepath, 'r') as f:
            sentences = self.preprocess([i.strip() for i in f])
            self.vocab_size = len(self.word_counts)
            self.word_to_idx = {i: o for o,
                                i in enumerate(self.word_counts.keys())}
            self.idx_to_word = [i for i in self.word_to_idx.keys()]
            for i, s in enumerate(sentences):
                s = [self.word_to_idx[i] for i in s]
                doc = Document(i, s)
                self.docs.append(doc)
            return sentences

    def preprocess(self, sentences):
        words = [j for i in sentences for j in nltk.word_tokenize(i)]
        self.word_counts = collections.Counter(words)
        stopwords = set(nltk.corpus.stopwords.words('english'))
        for i in self.word_counts.most_common()[::-1]:
            if i[1] < self.floor:
                stopwords.add(i[0])
        for i, s in enumerate(sentences):
            sentences[i] = [i for i in s.split(' ') if i not in stopwords]
            self.total_words += len(sentences[i])
        self.word_counts = collections.Counter(
            [j for i in sentences for j in i])
        return [i for i in sentences if len(i) > 0]

    def collocations(self, corpus, window_size=10):
        """Build the co-occurence matrix"""
        vocab_size = len(self.word_counts)
        self.word_to_idx = {o: i for i,
                            o in enumerate(self.word_counts.keys())}
        shape = (vocab_size, vocab_size)
        print('Starting co-occurence matrix build...')
        cooccurences = dok_matrix(shape)
        stopwords = [i[0]
                     for i in self.word_counts.most_common(self.stopword_k)]
        self.word_counts = self.word_counts - \
            Counter(self.word_counts.most_common(self.stopword_k))
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
                        cooccurences[a, b] += 1

        reciprocal = 1/sum(self.word_counts.values())

        self.idx_to_word = list(self.word_to_idx.keys())
        print('Computing PPMI...')
        ppmi = dok_matrix(shape)
        total = np.sum(cooccurences)
        for i, j in zip(np.nonzero(cooccurences)[0], np.nonzero(cooccurences)[1]):
            index_i = self.idx_to_word[i]
            index_j = self.idx_to_word[j]
            frequency = cooccurences[i, j]
            joint_probability = frequency / total
            probability_i = (frequency * self.word_counts[index_i]) / total
            probability_j = (frequency * self.word_counts[index_j]) / total
            denominator = probability_i * probability_j
            if denominator > 0:
                pmi = math.log(joint_probability /
                               (probability_i * probability_j), 2)
                ppmi[i, j] = max(0, pmi)
            else:
                ppmi[i, j] = 0
        print('finished computing')
        self.collocations = cooccurences
        self.shape = cooccurences.shape
        self.senses = [None] * self.shape[0]

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
