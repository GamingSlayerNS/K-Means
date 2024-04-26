# This is an implementation of K-Means clustering algorithm. Naxel Santiago & Danny Bao
# Press Shift+F10 to execute it in PyCharm.

import math
import random
import sys
import re
import numpy as np
import pandas

# fetch dataset
filePath = './Health-Tweets/nytimeshealth.txt'
with open(filePath, 'r', encoding='utf-8') as file:
    tweets = [line.strip() for line in file]


class KMeans:
    def __init__(self) -> None:
        pass

    def preprocessTweet(self, tweet):
        # Remove the tweet id and timestamp
        tweet = tweet.split('|', 2)[-1]

        # Remove any word that starts with the symbol @
        tweet = re.sub(r'@\w+', '', tweet)

        # Remove any hashtag symbols
        tweet = tweet.replace('#', '')

        # Remove any URL
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)

        # Convert every word to lowercase
        tweet = tweet.lower()

        return tweet.strip()

    # Calculate the jaccard distance
    def jaccardDistance(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return 1 - (intersection / union)

    def assignCentroid(self, ):
        pass

    def updateCentroids(self, clusters):
        pass

    def runKMeans(self, tweets, k, maxIterations=100):
        initialCentroids = random.sample(tweets, k)

# Press the green button in the gutter to run 
if __name__ == '__main__':
    kmeans = KMeans()
    preprocessedTweets = [kmeans.preprocessTweet(tweet) for tweet in tweets]

    print(f"Imported {len(preprocessedTweets)} tweets.\n")
    print('\n'.join(preprocessedTweets[:50]))

