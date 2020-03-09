import nltk
import twint
import pandas as pd
import re
import datetime


def partition():
    tweets = pd.read_csv('tweets.csv')
    tweets = tweets.loc[:, tweets.columns.intersection(
        ['date', 'tweet'])]
    tweets = tweets.sort_values(by='date')
    old = tweets[:int(len(tweets) * 0.6)]
    new = tweets[int(len(tweets) * 0.6):]
    old.to_csv('reference.csv')
    new.to_csv('focus.csv')


def scrape():
    config = twint.Config()
    config.Output = "tweets.csv"
    config.Store_csv = True
    config.Limit = 100
    config.Lang = "en"
    curr = datetime.datetime(2020, 3, 1)
    end = datetime.datetime(2006, 4, 1)
    stopwords = nltk.corpus.stopwords.words('english')
    for i in stopwords:
        while curr > end:
            until = curr.strftime("%Y-%m-%d")
            curr -= datetime.timedelta(days=1)
            since = curr.strftime("%Y-%m-%d")
            config.Since = since
            config.Until = until
            config.Search = i
            twint.run.Search(config)
        curr = datetime.datetime(2020, 3, 1)


def process(filepath):
    data = pd.read_csv(filepath)
    text = list(data.tweet.values)
    cleaned = map(lambda x: re.sub('[@#\n]|pic.twitter.com/.*|', '', x), text)
    write = "\n".join(cleaned)
    with open(filepath[:-4]+".txt", 'w') as f:
        f.write(write)


if __name__ == '__main__':
    scrape()
