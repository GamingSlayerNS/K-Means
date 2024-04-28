# Twitter K-Means Clustering Algorithm

This project implements a K-Means clustering algorithm from scratch to analyze and cluster Twitter tweets. It aims to find patterns and group similar tweets using the Jaccard distance as the similarity metric. The project is designed to run with tweets from the UCI Twitter dataset, focusing on exploring thematic consistencies or variations across tweets.

## Project Structure

- `main.py`: Contains the main K-Means clustering algorithm and data preprocessing methods.
- `Health-Tweets/`: Directory containing the dataset file.
- `README.md`: This file, providing an overview and instructions for the project.

## Prerequisites

Before running this project, you need to have Python installed on your machine (Python 3.10 recommended). Additionally, the following Python packages are required:

- `random`
- `re` (part of the standard library)

You can install the necessary packages using pip:

```bash
pip install numpy pandas
```

## Dataset
The dataset used is the UCI Twitter dataset, which can be found at the UCI Machine Learning Repository. The data needs to be downloaded and placed in the data/ directory.

## Usage

1. Configuration: You can modify the parameters in the KMeans.py script to fit the number of clusters (k), the range of data to process (data_range), and the maximum iterations (max_iterations).
2. Running the script: Execute the script from your command line by navigating to the project directory and running:

## Authors

- Naxel Santiago
- Danny Bao