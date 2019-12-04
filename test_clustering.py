from data import Corpus

matrix = Corpus('trial_data_public/corpora/english/corpus1/corpus1.txt')
print('Getting clusters for "walk"')
matrix.get_clusters('walk')
print('Getting clusters for "god"')
matrix.get_clusters('god')
print('Getting clusters for "distance"')
matrix.get_clusters('distance')
print('Getting clusters for "small"')
matrix.get_clusters('distance')
