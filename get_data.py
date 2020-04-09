"""
Scripts for getting Twitter data
"""
import nltk
import twint
import pandas as pd
import re
import datetime


def partition():
    """
    Read tweet data from a CSV and partition into reference and focus corpora by date
    """
    tweets = pd.read_csv('tweets.csv')
    tweets = tweets[:500000]
    tweets = tweets.loc[:, tweets.columns.intersection(
        ['date', 'tweet'])]
    tweets = tweets.sort_values(by='date')
    tweets = tweets.tweet
    old = tweets[:int(len(tweets) * 0.7)]
    new = tweets[int(len(tweets) * 0.7):]
    process(old, 'reference.txt')
    process(new, 'focus.txt')


def scrape():
    """
    Retrieve data from Twitter and save to a CSV
    """
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
        print(f'Stopword complete: {i}')
        curr = datetime.datetime(2020, 3, 1)


def process(data, filepath):
    """
    Remove links, handles, newlines and '#'s from tweet data and write to .txt

    Parameters
    ----------
    data: pandas Series of tweets
    filepath: location to save processed tweets to
    """
    text = list(data.tweet.values)
    cleaned = map(lambda x: re.sub(
        '[#\n]|pic.twitter.com/.*|@\S*|http(s)?://\S*', '', x), text)
    write = "\n".join(cleaned)
    with open(filepath, 'w') as f:
        f.write(write)


# run the scripts
if __name__ == '__main__':
    scrape()
    partition()
