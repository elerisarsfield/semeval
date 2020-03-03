import nltk
import twint
import pandas as pd


def main():
    config = twint.Config
    config.Output = "tweets.csv"
    config.Store_csv = True
    config.Pandas = True
    config.Lang = "en"

    stopwords = nltk.corpus.stopwords.words('english')
    for i in stopwords:
        config.Search = i
        twint.run.Search(config)

    tweets = pd.read_csv('tweets.csv')


if __name__ == '__main__':
    main()
