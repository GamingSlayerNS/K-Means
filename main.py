# This is an implementation of a simple Neural Network. Naxel Santiago & Danny Bao
# Press Shift+F10 to execute it in PyCharm.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
import sys

import numpy as np
import pandas

# fetch dataset
filePath = './Health-Tweets/nytimeshealth.txt'
with open(filePath, 'r', encoding='utf-8') as file:
    tweets = [line.strip() for line in file]


class KMeans:
    def __init__(self) -> None:
        pass


# Press the green button in the gutter to run the Neural Network.
if __name__ == '__main__':
    print(tweets)

