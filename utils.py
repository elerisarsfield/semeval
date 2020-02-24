def get_targets(location):
    with open(location, 'r') as f:
        return [i.strip() for i in f]
