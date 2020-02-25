# Runs semeval eval, assuming test data is in the same folder as the project
python main.py 'test_data_public/english/corpus1/lemma/ccoha1.txt' 'test_data_public/english/corpus2/lemma/ccoha2.txt' 'test_data_public/english/targets.txt' 'semeval/answer' --semeval_mode=True;
# Run inference on a set of data in the same folder as the project
python main.py 'data/corpus1.txt' 'data/corpus2.txt' 'answer'
