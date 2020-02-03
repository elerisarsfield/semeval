import nltk
import random
import math
from scipy import stats


class Document():
    def __init__(self, idx, doc):
        self.idx = idx
        self.partition = []
        self.topics = []
        self.topic_to_global_idx = []
        self.global_to_local_topic = dict()
        self.words = nltk.word_tokenize(doc)

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


class HDP():
    def __init__(self, vocab_size, eta=0.1, alpha=1, gamma=1):
        self.eta = eta
        self.alpha = alpha
        self.gamma = gamma
        self.senses = []
        self.pi = []
        self.pi_left = 1
        self.beta = []
        self.docs = []
        self.smoothing = []

    def init_partition(self, documents):
        for i, d in enumerate(documents):
            doc = Document(i, d)
            doc.init_partition(self.alpha)
            if len(doc.words) > 0:
                self.docs.append(doc)
        N = 0
        for i, d in enumerate(self.docs):
            for j in range(len(d.partition)):
                N += 1
                prior = [0] * len(self.senses)
                for k in range(len(self.senses)):
                    probability = len(self.senses[k])/(N+self.gamma-1)
                    prior[k] = probability
                new = self.gamma/(N+self.gamma-1)
                prior.append(new)
                sense = random.random()
                if sense > sum(prior[:-1]):
                    self.senses.append([(i, j)])
#                    d.topics.append([j])
                    d.topic_to_global_idx.append(
                        len(self.senses)-1)
                    d.global_to_local_topic[len(
                        self.senses)-1] = len(d.topics) - 1
#                    print(d.global_to_local_topic)
                else:
                    curr = 0
                    for k, p in enumerate(prior):
                        curr += p
                        if curr > sense:
                            self.senses[k].append((i, j))
                            if k not in d.topic_to_global_idx:
                                d.topic_to_global_idx.append(k)
#                            print(d.topics)
 #                           d.topics[d.global_to_local_topic[k]].append(j)
                            break
        self.pi = [1] * len(self.senses)
        self.beta = [1] * len(self.senses)
        self.smoothing = [1] * len(documents)

    def gibbs(self, permute=True, remove=True):
        if permute:
            pass
        self.sample_top()
        change = 0
        for j, d in enumerate(self.docs):
            for i in range(len(d.words)):
                change += self.sample_word(d, i, remove)

    def sample_word(self):
        # choose tables
        j_1 = random.randint(0, len(self.docs))
        t_1 = random.randint(0, len(self.docs[j_1].partition) - 1)
        j_2 = random.randint(0, len(self.docs))
        t_2 = random.randint(0, len(self.docs[j_2].partition) - 1)
        kj1t1 = self.docs[j_1].topic_to_global_idx[t_1]
        kj2t2 = self.docs[j_2].topic_to_global_idx[t_2]

        # split case
        if kj1tl == kj2t2:
            all_others = list(filter(lambda x: x != (j_1, t_1)
                                     and x != (j_2, t_2), self.senses[kj1t1]))
            assignments = [None] * len(all_others)

        # merge case
else:
    pass

    def sample_top(self):
        """Run Gibbs sampling with split-merge operations"""
        total = 0
        for k in range(len(self.senses)):
            self.pi[k] = stats.gamma(self.beta[k])
            total += self.pi[k]
        self.pi_left = stats.gamma(self.gamma)
        total += self.pi_left
        self.pi = [k/total for k in self.pi]
        self.pi_left /= total
        eta = self.vocab_size * self.eta
#        smoothing_sum = 0
#        for k in range(len(self.senses)):
#            self.smoothing[k] = self.alpha * self.pi[k] / (self.)

    def llikelihood(self):
        likelihood = 0
        likelihood += len(self.docs) * math.lgamma(self.alpha)
