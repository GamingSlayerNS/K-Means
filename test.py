import random
import math
import random
import sys
import re
import numpy as np
import pandas

# fetch dataset
filePath = './Health-Tweets/nytimeshealth.txt'
with open(filePath, 'r', encoding='utf-8') as file:
    raw_tweets = [line.strip() for line in file]

    def preprocessTweet(tweet):
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

# Function to calculate Jaccard distance
def jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (intersection / union)


# Function to assign tweets to the nearest centroids
def assign_tweets_to_centroids(tweets, centroids):
    clusters = {}
    for i in range(len(centroids)):
        clusters[i] = []

    for tweet in tweets:
        distances = [jaccard_distance(tweet, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(tweet)

    return clusters


# Function to update centroids
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters.values():
        # Combine all sets in the cluster and find the average set
        cluster_union = set().union(*cluster)
        # The new centroid is the set that has the smallest average Jaccard distance to all sets in the cluster
        new_centroid = min(cluster,
                           key=lambda tweet: sum(jaccard_distance(tweet, other) for other in cluster) / len(cluster))
        new_centroids.append(new_centroid)
    return new_centroids


# K-Means clustering
def k_means(tweets, k, max_iterations=100):
    # Randomly select initial centroids
    initial_centroids = random.sample(tweets, k)

    centroids = initial_centroids
    clusters = {}

    for iteration in range(max_iterations):
        clusters = assign_tweets_to_centroids(tweets, centroids)
        new_centroids = update_centroids(clusters)

        # If centroids don't change, break out of the loop
        if set(map(tuple, new_centroids)) == set(map(tuple, centroids)):
            break
        centroids = new_centroids

    return clusters


# Example usage
raw_tweets = [preprocessTweet(tweet) for tweet in raw_tweets[:10]]
print("raw tweets:\n", raw_tweets)
tweets = [set(tweet.split()) for tweet in raw_tweets]  # raw_tweets is your list of tweets
print("preprocessed tweets:\n", tweets)
print("Running Kmeans")
clusters = k_means(tweets, 5)

# Display the results
for cluster_index, tweet_sets in clusters.items():
    print(f"Cluster {cluster_index}: {[' '.join(tweet_set) for tweet_set in tweet_sets]}")
