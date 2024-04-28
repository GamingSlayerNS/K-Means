# This is an implementation of K-Means clustering algorithm. Naxel Santiago & Danny Bao
# Press Shift+F10 to execute it in PyCharm.

import random
import re

# fetch dataset
filePath = './Health-Tweets/nytimeshealth.txt'
with open(filePath, 'r', encoding='utf-8') as file:
    raw_tweets = [line.strip() for line in file]


class KMeans:
    def __init__(self, k, data_range, max_iterations):
        self.k = k
        self.data_range = data_range
        self.max_iterations = max_iterations
        self.preprocessed_tweets = None
        self.tweet_sets = None
        self.clusters = None
        self.centroids = None

    def preprocessTweet(self, raw_data):
        tweets = []
        for tweet in raw_data[:100]:
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
            tweet = tweet.strip()

            tweets += [tweet]

        self.preprocessed_tweets = tweets
        self.tweet_sets = [set(tweet.split()) for tweet in self.preprocessed_tweets]

    # Calculate the jaccard distance
    @staticmethod
    def jaccardDistance(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return 1 - (intersection / union)

    def assignCentroid(self, centroids):
        clusters = {i: [] for i in range(len(centroids))}
        for index, tweet in enumerate(self.tweet_sets):
            distances = [self.jaccardDistance(tweet, centroid) for centroid in centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(index)
        return clusters

    def updateCentroids(self, clusters):
        new_centroids = []
        for cluster in clusters.values():
            # Find the centroid as the tweet with the smallest average Jaccard distance to others in the cluster
            if cluster:  # Checking if the cluster is not empty
                new_centroid = min(cluster, key=lambda idx: sum(
                    self.jaccardDistance(
                        self.tweet_sets[idx], self.tweet_sets[other_idx]) for other_idx in cluster) / len(cluster)
                    )
                new_centroids.append(self.tweet_sets[new_centroid])
            else:
                new_centroids.append(set())  # Just in case there's an empty cluster
        return new_centroids

    # Execute K-Means Clustering
    def runKMeans(self):
        print("\nRunning K-Means...")
        initial_centroids = random.sample(self.tweet_sets, self.k)

        centroids = initial_centroids
        clusters = {}

        for iteration in range(self.max_iterations):
            self.clusters = self.assignCentroid(centroids)
            new_centroids = self.updateCentroids(self.clusters)

            # If centroids don't change, break out of the loop
            if set(map(tuple, new_centroids)) == set(map(tuple, centroids)):
                break
            self.centroids = new_centroids

        return clusters, centroids

    # Calculate Sum of Square Errors
    def calculateSse(self, clusters, tweets, centroids):
        sse_per_cluster = {}
        total_sse = 0  # Initialize total SSE to zero

        for cluster_index, tweet_indices in clusters.items():
            centroid = centroids[cluster_index]
            # Calculate the SSE for the cluster
            cluster_sse = sum(self.jaccardDistance(tweets[idx], centroid) ** 2 for idx in tweet_indices)
            sse_per_cluster[cluster_index] = cluster_sse
            total_sse += cluster_sse  # Add the cluster's SSE to the total SSE

        return sse_per_cluster, total_sse

    def renderClusters(self):
        # Display the results
        for cluster_index, tweet_indices in self.clusters.items():
            print(f"Cluster {cluster_index + 1}: {[self.preprocessed_tweets[idx] for idx in tweet_indices]}\n")

        for cluster_index, tweet_indices in self.clusters.items():
            print(f"Size of cluster {cluster_index + 1}: {len(tweet_indices)}")

        sse_per_cluster, total_sse = self.calculateSse(self.clusters, self.tweet_sets, self.centroids)
        for cluster_index, sse in sse_per_cluster.items():
            print(f"SSE for Cluster {cluster_index + 1}: {sse}")
        print(f"Total Sum of Square Error: {total_sse}")


# Press the green button in the gutter to run 
if __name__ == '__main__':
    kmeans = KMeans(k=10, data_range=100, max_iterations=100)

    kmeans.preprocessTweet(raw_tweets)
    print("preprocessed tweets:\n", kmeans.preprocessed_tweets)
    print("tweet sets:\n", kmeans.tweet_sets)
    kmeans.runKMeans()
    kmeans.renderClusters()
