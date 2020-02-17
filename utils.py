from scipy.stats import uniform, gamma


def density(word, sense):
    pass


def metropolis_hastings(prior, likelihood, transition):
    """Determine whether a split or merge is accepted"""
    probabilities = prior * likelihood
    acceptance = probabilities * transition
    u = uniform.rvs(0, 1)
    return True if u < acceptance else False


def cond_density(words_by_topic, words_by_table,
                 vocab_size, eta):
    adjusted_vocab = vocab_size * eta
    table_size = len(words_by_table)
    adjusted_sense_size = len(words_by_topic - words_by_table)
    density = gamma(adjusted_sense_size + adjusted_vocab).rvs() / \
        gamma(adjusted_sense_size + adjusted_vocab + table_size).rvs()
    return density


def cond_likelihood(topic, vocab_size, eta):
    pass
