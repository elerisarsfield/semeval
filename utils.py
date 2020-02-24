def get_targets(location):
    """Retrieve a list of target words with corresponding POS"""
    with open(location, 'r') as f:
        targets = [i.strip().split('_') for i in f]
        return list(map(lambda x: x[0], targets))
        
