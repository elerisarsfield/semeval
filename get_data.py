import twitterscraper

def main():
    tweets = twitterscraper.query_tweets("")
    with open("tweets.json","w") as f:
        for tweet in tweets:
            f.write(tweet.encode('utf-8'))
        

if __name__ == '__main__':
    main()
