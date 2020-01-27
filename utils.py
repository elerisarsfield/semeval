from scipy.stats import uniform


def metropolis_hastings(prior, likelihood, transition):
    """Determine whether a split or merge is accepted"""
    probabilities = prior * likelihood
    acceptance = probabilities * transition
    u = uniform.rvs(0, 1)
    return True if u < acceptance else False
