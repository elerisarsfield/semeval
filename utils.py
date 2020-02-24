def get_targets(location):
    """Retrieve a list of target words with corresponding POS"""
    with open(location, 'r') as f:
        return [tuple(i.strip().split('_')) for i in f]
        
