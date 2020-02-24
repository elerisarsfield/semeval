import random
import math
import numpy as np
import os
import pickle
from scipy import stats


class HDP():
    def __init__(self, vocab_size, output, eta=0.1, alpha=1, gamma=1):
        self.eta = eta
        self.alpha = alpha
        self.gamma = gamma
        self.senses = []
        self.sense_indices = []
        self.sense_counts = []
        self.pi = []
        self.vocab_size = vocab_size
        self.beta = []
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

#                    d.topics.append([j])
                    d.topic_to_global_idx.append(
                        len(self.sense_indices)-1)
                    d.global_to_local_topic[len(
                        self.sense_indices)-1] = len(d.topics) - 1
#                    print(d.global_to_local_topic)
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
        self.pi = [1] * len(self.senses)
        self.beta = np.random.gamma(
            1.0, 1.0, (len(self.senses), self.vocab_size)) * (
                len(documents) / self.vocab_size)
        self.smoothing = [1] * len(documents)
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
                        self.senses[global_curr][i] -= 1
                        self.senses[t][i] += 1
                        j.topics[global_curr] -= self.eta
                        j.topics[t] += self.eta
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
                            j.topics[global_curr] -= self.eta
                            j.topics[topic] += self.eta
                            self.senses[global_curr][i] -= 1
                            self.senses[topic][i] += 1
                            self.sense_indices[topic].append(
                                (j.idx, len(j.partition) - 1))
                    else:
                        j.topics[global_curr] -= self.eta
                        j.topics = np.append(j.topics, self.eta)
                        self.senses[global_curr][i] -= 1
                        new = np.zeros((1, self.senses.shape[1]))
                        new[0][i] = 1
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

    def gibbs(self, permute=True, remove=True):
        if permute:
            random.shuffle(self.docs)
            for d in self.docs:
                random.shuffle(d.words)
        self.sample_top()
        change = 0
        for j, d in enumerate(self.docs):
            for i in range(len(d.words)):
                change += self.sample_word(d, i, remove)
            self.sample_table(d)
            self.sample_top()
        return change

    def sample_word(self, doc, word_index, remove):
        if remove:
            k = doc.topics[word_index]
            update = doc.counts[doc.words[word_index]]
            1 if remove else doc.counts[doc.words[word_index]]
            self.smoothing_sum -= self.smoothing[k]
            self.sense_counts[k] += update

    def split_merge(self):
        # choose tables
        j_1 = random.randint(0, len(self.docs) - 1)
        t_1 = random.randint(0, len(self.docs[j_1].partition) - 1)
        j_2 = random.randint(0, len(self.docs) - 1)
        t_2 = random.randint(0, len(self.docs[j_2].partition) - 1)
        kj1t1 = self.docs[j_1].topic_to_global_idx[t_1]
        kj2t2 = self.docs[j_2].topic_to_global_idx[t_2]

        # split case
        if kj1t1 == kj2t2:
            all_others = list(filter(lambda x: x != (j_1, t_1) and x != (
                j_2, t_2), self.sense_indices[kj1t1]))
            random.shuffle(all_others)
            k1 = []
            k1_words = set()
            k2 = []
            k2_words = set()
            for i in all_others:
                j = i[0]
                t = i[1]
                k1_density = cond_density(k2_words, self.docs[j].partition[t],
                                          self.vocab_size, self.eta)
                k2_density = cond_density(k2_words, self.docs[j].partition[t],
                                          self.vocab_size, self.eta)
                k1_p = ((len(k1) * k1_density) + 1)/2
                k2_p = ((len(k2) * k2_density) + 1)/2
                k = random.random()
                if k <= k1_p:
                    k1.append((j, t))
                    k1_words.update(self.docs[j].partition[t])
                else:
                    k2.append((j, t))
                    k2_words.update(self.docs[j].partition[t])
            prior = self.gamma*(
                ((math.factorial(len(k1) - 1))*(math.factorial(len(k2) - 1)))
                / (math.factorial(len(k1)+len(k2)-1)))
            likelihood_k1 = cond_likelihood(
                k1_words, self.vocab_size, self.eta)
            likelihood_k2 = cond_likelihood(
                k2_words, self.vocab_size, self.eta)
            likelihood_k = cond_likelihood(
                k1_words + k2_words, self.vocab_size, self.eta)
            likelihood = (likelihood_k1 * likelihood_k2)/likelihood_k
            transition = 1
            for i in k1:
                transition *= len(k1) * cond_density(
                    k1_words, self.docs[i[0]].partition[i[1]],
                    self.vocab_size, self.eta)
            for i in k2:
                transition *= len(k2) * cond_density(
                    k2_words, self.docs[i[0]].partition[i[1]],
                    self.vocab_size, self.eta)
            accept = metropolis_hastings(prior, likelihood, 1/transition)
            if accept:
                del self.sense_indices[kj1t1]
                self.sense_indices.append(k1)
                self.sense_indices.append(k2)
                del self.senses[kj1t1]
                self.senses.append(list(k1_words))
                self.senses.append(list(k2_words))
                for i in k1:
                    self.docs[i[0]].topics = [
                        len(self.senses) - 2 if j == kj1t1 else j]
                for i in k2:
                    self.docs[i[0]].topics = [
                        len(self.senses) - 1 if j == kj1t1 else j]
        # merge case
        else:
            all_others = list(filter(lambda x: x != (j_1, t_1) and x != (
                j_2, t_2), self.sense_indices[kj1t1] + self.sense_indices[kj2t2]))
            random.shuffle(all_others)
            k1 = list(filter(lambda x: x != (j_1, t_1),
                             self.sense_indices[kj1t1]))
            k1_words = set([self.docs[i[0]].partition[i[1]] for i in k1])
            k2 = list(filter(lambda x: x != (j_2, t_2),
                             self.sense_indices[kj2t2]))
            k2_words = set([self.docs[i[0]].partition[i[1]] for i in k2])
            k_words = k1_words + k2_words
            prior = self.gamma*(
                ((math.factorial(len(k1) - 1))*(math.factorial(len(k2) - 1)))
                / (math.factorial(len(k1)+len(k2)-1)))
            likelihood_k1 = cond_likelihood(
                k1_words, self.vocab_size, self.eta)
            likelihood_k2 = cond_likelihood(
                k2_words, self.vocab_size, self.eta)
            likelihood_k = cond_likelihood(
                k1_words + k2_words, self.vocab_size, self.eta)
            likelihood = likelihood_k/(likelihood_k1 * likelihood_k2)
            transition = 1
            for i in k1:
                transition *= len(k1) * cond_density(
                    k1_words, self.docs[i[0]].partition[i[1]],
                    self.vocab_size, self.eta)
            for i in k2:
                transition *= len(k2) * cond_density(
                    k2_words, self.docs[i[0]].partition[i[1]],
                    self.vocab_size, self.eta)
            accept = metropolis_hastings(prior, likelihood, transition)
            if accept:
                del self.senses[kj1t1]
                del self.senses[kj2t2]
                self.senses.append(k_words)
                del self.sense_indices[kj1t1]
                del self.sense_indices[kj2t2]
                self.sense_indices.append(all_others)
                for i in k1 + k2:
                    self.docs[i[0]].topic_to_global_idx[i[1]] = len(
                        self.senses) - 1

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
        smoothing_sum = 0
        for k in range(len(self.senses)):
            self.smoothing[k] = self.alpha * self.pi[k]

    def save(self, it):
        filename = 'senses_'+str(it)+'.pkl'
        out = os.path.join(self.output, filename)
        pickle.dump(self, open(out, 'wb'))
