import random
import numpy as np
import os
import pickle


class HDP():
    def __init__(self, vocab_size, output, eta=0.1, alpha=1, gamma=1):
        self.alpha = alpha
        self.gamma = gamma
        self.senses = []
        self.sense_indices = []
        self.vocab_size = vocab_size
        self.output = output

    def init_partition(self, documents):
        N = 0
        senses = []
        for i, d in enumerate(documents):
            for j in range(len(d.partition)):
                N += 1
                prior = [0] * len(self.sense_indices)
                for k in range(len(self.sense_indices)):
                    probability = len(self.sense_indices[k])/(N+self.gamma-1)
                    prior[k] = probability
                new = self.gamma/(N+self.gamma-1)
                prior.append(new)
                sense = random.random()
                if sense > sum(prior[:-1]):

                    self.sense_indices.append([(i, j)])
                    senses.append(np.fromiter(
                        (1 if i in d.partition[j] else 0
                         for i in range(self.vocab_size)), dtype=np.float32,
                        count=self.vocab_size))
                    d.topic_to_global_idx.append(
                        len(self.sense_indices)-1)
                else:
                    curr = 0
                    for k, p in enumerate(prior):
                        curr += p
                        if curr > sense:
                            self.sense_indices[k].append((i, j))
                            d.topic_to_global_idx.append(k)
                            senses[k] = senses[k] + np.fromiter((
                                1 if i in d.partition[j] else 0 for i in range(
                                    self.vocab_size)), dtype=np.float32,
                                count=self.vocab_size)
                            break
        self.senses = np.array(senses)

    def sample_table(self, j, i, ppmi):
        x = j.words[i]
        total_assigned = np.count_nonzero(self.senses)
        prior_d = ppmi.sum() * (self.gamma/(self.gamma + total_assigned))
        cond = np.zeros((len(j.partition) + 1,))
        new_cond_p = 0
        curr, global_curr = -1, -1
        for t, p in enumerate(j.partition):
            size = len(p)
            if x in p:
                size -= 1
                curr = t
                global_curr = j.topic_to_global_idx[curr]
            k = j.topic_to_global_idx[t]
            topic_word_density = self.senses[k] * ppmi.T
            cond[t] = size * topic_word_density
            new_cond_p += (np.count_nonzero(self.senses[k])/(
                total_assigned + self.gamma)) * topic_word_density
        new_cond_p += prior_d
        new_cond_p *= self.alpha
        cond[-1] = new_cond_p
        if cond.sum() > 0:
            cond /= cond.sum()
        else:
            return
        sample = random.random()
        pos = 0
        for t, v in enumerate(cond):
            pos += v
            if v > sample:
                # if assigning to current table just return
                if t == global_curr:
                    return
                # reassign to a different table
                elif t < len(j.partition):
                    new = j.topic_to_global_idx[t]
                    if new != global_curr:
                        self.senses[global_curr][x] -= 1
                        self.senses[t][x] += 1
                    j.partition[curr].remove(x)
                    j.partition[t].append(x)
                # create a new table
                else:
                    topic = self.sample_topic(ppmi)
                    j.partition[curr].remove(x)
                    j.partition.append([x])
                    j.topic_to_global_idx.append(topic)
                    if topic < len(self.senses):
                        if topic != global_curr:
                            self.senses[global_curr][x] -= 1
                            self.senses[topic][x] += 1
                            self.sense_indices[topic].append(
                                (j.idx, len(j.partition) - 1))
                    else:
                        self.senses[global_curr][x] -= 1
                        new = np.zeros((1, self.senses.shape[1]))
                        new[0][x] = 1
                        self.senses = np.concatenate((self.senses, new))
                        self.sense_indices.append(
                            (j.idx, len(j.partition) - 1))
                if not all(j.partition):
                    t = j.partition.index([])
                    del j.partition[t]
                    k = j.topic_to_global_idx.pop(t)
                    if len(self.sense_indices[k]) == 0:
                        self.senses[k] = np.zeros(self.senses[k].shape)
                break

    def sample_topic(self, ppmi):
        cond = np.zeros((self.senses.shape[0] + 1,))
        for k in range(self.senses.shape[0]):
            size = np.count_nonzero(self.senses[k])
            topic_word_density = self.senses[k] * ppmi.T
            cond[k] = size * topic_word_density
        cond[-1] = ppmi.sum() * self.gamma
        assert cond.sum() > 0
        cond /= cond.sum()
        sample = random.random()
        curr = 0
        for k, p in enumerate(cond):
            curr += p
            if curr > sample:
                return k
        return len(cond) - 1

    def save(self, it):
        filename = 'senses_'+str(it)+'.pkl'
        out = os.path.join(self.output, filename)
        pickle.dump(self, open(out, 'wb'))
