import data
START_CORPUS = 'trial_data_public/corpora/english/corpus1/corpus1.txt'

def main():
    sentences, counts = data.split_words(START_CORPUS)
    cooccurence = data.collocations(sentences, counts)

if '__name__' == '__main__':
    main()
