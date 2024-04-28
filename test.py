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
    clusters = {i: [] for i in range(len(centroids))}
    for index, tweet in enumerate(tweets):
        distances = [jaccard_distance(tweet, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(index)  # Storing index instead of tweet content
    return clusters


# Function to update centroids
def update_centroids(clusters, tweets):
    new_centroids = []
    for cluster in clusters.values():
        # Find the centroid as the tweet with the smallest average Jaccard distance to others in the cluster
        if cluster:  # Checking if the cluster is not empty
            new_centroid = min(cluster, key=lambda idx: sum(jaccard_distance(tweets[idx], tweets[other_idx]) for other_idx in cluster) / len(cluster))
            new_centroids.append(tweets[new_centroid])
        else:
            new_centroids.append(set())  # Just in case there's an empty cluster
    return new_centroids


# K-Means clustering
def k_means(tweets, k, max_iterations=100):
    # Randomly select initial centroids
    initial_centroids = random.sample(tweets, k)

    centroids = initial_centroids
    clusters = {}

    for iteration in range(max_iterations):
        clusters = assign_tweets_to_centroids(tweets, centroids)
        new_centroids = update_centroids(clusters, tweets)

        # If centroids don't change, break out of the loop
        if set(map(tuple, new_centroids)) == set(map(tuple, centroids)):
            break
        centroids = new_centroids

    return clusters, centroids


# Calculate Sum of Square Errors
def calculate_sse(clusters, tweets, centroids):
    sse_per_cluster = {}
    total_sse = 0  # Initialize total SSE to zero

    for cluster_index, tweet_indices in clusters.items():
        centroid = centroids[cluster_index]
        # Calculate the SSE for the cluster
        cluster_sse = sum(jaccard_distance(tweets[idx], centroid)**2 for idx in tweet_indices)
        sse_per_cluster[cluster_index] = cluster_sse
        total_sse += cluster_sse  # Add the cluster's SSE to the total SSE

    return sse_per_cluster, total_sse


# Example usage
preprocessed_tweets = [preprocessTweet(tweet) for tweet in raw_tweets[:6402]]
print("preprocessed tweets:\n", preprocessed_tweets)
tweet_sets = [set(tweet.split()) for tweet in preprocessed_tweets]  # raw_tweets is your list of tweets
print("tweet sets:\n", tweet_sets)
print("Running Kmeans")
clusters, centroids = k_means(tweet_sets, 50)
print(f"Centroids: {centroids}")

# Display the results
for cluster_index, tweet_indices in clusters.items():
    print(f"Cluster {cluster_index+1}: {[preprocessed_tweets[idx] for idx in tweet_indices]}")

for cluster_index, tweet_indices in clusters.items():
    print(f"Size of cluster {cluster_index+1}: {len(tweet_indices)}")

sse_per_cluster, total_sse = calculate_sse(clusters, tweet_sets, centroids)
for cluster_index, sse in sse_per_cluster.items():
    print(f"SSE for Cluster {cluster_index+1}: {sse}")
print(f"Total Sum of Square Error: {total_sse}")
