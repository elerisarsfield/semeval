import nltk
import random


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
        self.beta = []
        self.docs = []

    def init_partition(self, documents):
        for i, d in enumerate(documents):
            doc = Document(i, d)
            doc.init_partition(self.alpha)
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
                    d.topics.append([j])
                    d.topic_to_global_idx.append(
                        [len(self.senses)-1])
                    d.global_to_local_topic[len(
                        self.senses)-1] = len(d.topics) - 1
                    print(d.global_to_local_topic)
                else:
                    curr = 0
                    for k, p in enumerate(prior):
                        curr += p
                        if curr > sense:
                            self.senses[k].append((i, j))
#                            print(d.topics)
 #                           d.topics[d.global_to_local_topic[k]].append(j)
                            break

    def gibbs(self, permute=True, remove=True):
        if permute:
            pass
