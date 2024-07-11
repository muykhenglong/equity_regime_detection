# Regime Detection and Backtesting Investment Strategy

This project aims to detect market regimes using different models and backtest an investment strategy based on the detected regimes. The models used include Agglomerative Clustering, Gaussian Mixture Models (GMM), and Hidden Markov Models (HMM). The strategy involves taking long or short positions based on the predicted market regimes.

## Features

* Automated Data Fetching: Retrieves historical stock prices for the S&P 500 index.
* Data Preprocessing: Adds missing data for weekends and holidays using forward fill, and computes log returns of the moving average of closing prices.
* Regime Detection Models: Utilizes Agglomerative Clustering, GMM, and HMM to detect market regimes.
* Backtesting Strategy: Implements a trading strategy based on the detected regimes and evaluates its performance.
* Visualization: Plots the cumulative returns of the trading strategy compared to a buy-and-hold strategy.

## Requirements

Python 3.8.12

# Overciew
## Agglomerative Clustering Architecture

Agglomerative Clustering is a hierarchical clustering method that builds nested clusters in a bottom-up manner. This algorithm is particularly useful for identifying nested groups within data and is widely used in data mining and statistics.

The architecture of Agglomerative Clustering involves the following key steps:

1. Initialization:
Each data point starts as its own individual cluster. If there are n data points, there are initially n clusters.

2. Distance Calculation:
The distance between all pairs of clusters is calculated. Various distance metrics can be used, such as Euclidean distance, Manhattan distance, or others.

3. Merging Clusters:
* The two clusters that are closest to each other are merged into a single cluster.
* The distance matrix is updated to reflect the merge, and distances between the new cluster and all other clusters are recalculated.

4. Repeat:
Steps 2 and 3 are repeated until only one cluster remains, which contains all data points. The result is a hierarchy of clusters that can be represented as a dendrogram.

## Mathematics

The mathematical foundation of Agglomerative Clustering involves several key concepts:

1. Distance Metrics:
    - Euclidean Distance:
      $d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$
    - Manhattan Distance:
      $d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$
    
2. Linkage Criteria:
    - Determines how the distance between two clusters is computed based on the pairwise distances between points in the clusters. Common linkage criteria include:
    - Single Linkage: Distance between the closest points of the clusters.
      $d(C_i, C_j) = \min \{ d(x, y) : x \in C_i, y \in C_j \}$
    - Complete Linkage: Distance between the farthest points of the clusters.
      $d(C_i, C_j) = \max \{ d(x, y) : x \in C_i, y \in C_j \}$
    - Average Linkage: Average distance between all points of the clusters.
      $d(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$

3. Algorithm Steps:
    - Step 1: Initialize each data point as its own cluster.
    - Step 2: Compute the distance matrix for all pairs of clusters.
    - Step 3: Merge the two closest clusters according to the chosen linkage criterion.
    - Step 4: Update the distance matrix to reflect the new cluster configuration.
    - Step 5: Repeat steps 3 and 4 until a single cluster remains.

## Gaussian Mixture Models Architecture

Gaussian Mixture Models (GMM) are a probabilistic model for representing normally distributed subpopulations within an overall population. GMMs are widely used in clustering, density estimation, and pattern recognition.

The architecture of Gaussian Mixture Models involves the following key steps:

1. Initialization: Initialize the parameters of the Gaussian components, including means ($\mu$), covariances ($\Sigma$), and mixing coefficients ($\pi$).

2. Expectation-Maximization (EM) Algorithm:
    - Expectation Step (E-step):
        - Calculate the posterior probabilities (responsibilities) for each data point belonging to each Gaussian component.
    - Maximization Step (M-step):
        - Update the parameters ($\mu$), ($\Sigma$), ($\pi$) based on the current posterior probabilities.

3. Iteration:
    - Repeat the E-step and M-step until convergence, i.e., when the change in log-likelihood falls below a predefined threshold or the maximum number of iterations is reached.

### Mathematics

The mathematical foundation of Gaussian Mixture Models involves several key concepts:

1. Gaussian (Normal) Distribution:
    - The probability density function of a Gaussian distribution in $d$-dimensions is given by:
      $p(x|\mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$
      
2. Mixture Model:
    - The probability density function of a mixture of $K$ Gaussians is given by:
      $p(x|\lambda) = \sum_{k=1}^{K} \pi_k \cdot p(x|\mu_k, \Sigma_k)$
      where $\(\lambda = (\pi_k, \mu_k, \Sigma_k)\)$ are the parameters of the model.

3. Expectation-Maximization (EM) Algorithm:
    - E-step: Calculate the responsibility $\gamma_{ik}$ that component $k$ takes for data point \(i\):
      $\gamma_{ik} = \frac{\pi_k \cdot p(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \cdot p(x_i|\mu_j, \Sigma_j)}$
    - M-step: Update the parameters $\pi_k$, $\mu_k$, and $\Sigma_k$:
      $\pi_k = \frac{N_k}{N}$
      $\mu_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} x_i$
      $\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T$
      where $\(N_k = \sum_{i=1}^{N} \gamma_{ik}\)$ is the effective number of points assigned to component \(k\).

4. Log-Likelihood:
    - The log-likelihood of the data given the parameters $\lambda$ is given by:
      $\log L(\lambda) = \sum_{i=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \cdot p(x_i|\mu_k, \Sigma_k) \right)$

## Hidden Markov Models Architecture

Hidden Markov Models are statistical models that represent systems with unobservable (hidden) states. HMMs are widely used in time series analysis, speech recognition, and finance for modeling sequences with underlying hidden states.

The architecture of Hidden Markov Models involves the following key components:

1. States: A set of hidden states $S = \{S_1, S_2, \ldots, S_N\}$ that the system can be in. The states are not directly observable.

2. Observations: A set of observable outputs $O = \{O_1, O_2, \ldots, O_T\}$ corresponding to the states.

3. Transition Probabilities: The probabilities of transitioning from one state to another, represented by a matrix $A$, where $a_{ij} = P(S_{t+1} = S_j | S_t = S_i)$.

4. Emission Probabilities: The probabilities of observing a particular output from a state, represented by a matrix $B$, where $b_{j}(o_t) = P(O_t = o_t | S_t = S_j)$.

5. Initial State Probabilities: The probabilities of starting in each state, represented by a vector $\pi$, where $\pi_i = P(S_1 = S_i)$.

### Mathematics

The mathematical foundation of Hidden Markov Models involves several key algorithms:

1. Forward Algorithm: Computes the probability of an observed sequence given the model parameters.
      $\alpha_t(j) = P(O_1, O_2, \ldots, O_t, S_t = S_j | \lambda)$
      
2. Backward Algorithm: Computes the probability of the ending partial sequence given the model parameters.
      $\beta_t(i) = P(O_{t+1}, O_{t+2}, \ldots, O_T | S_t = S_i, \lambda)$
      
3. Baum-Welch Algorithm (EM Algorithm):
    - Estimates the model parameters $\lambda = (A, B, \pi)$ to maximize the likelihood of the observed sequence.
    - Expectation Step (E-step):
        - Calculate the forward and backward probabilities.
      $\gamma_t(i) = P(S_t = S_i | O, \lambda)$
      $\xi_t(i, j) = P(S_t = S_i, S_{t+1} = S_j | O, \lambda)$
    - Maximization Step (M-step):
        - Update the parameters $A, B, \pi$.
      $\pi_i = \gamma_1(i)$
      $a_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$
      $b_j(k) = \frac{\sum_{t=1}^{T} \gamma_t(j) \cdot 1(O_t = k)}{\sum_{t=1}^{T} \gamma_t(j)}$
      
4. Viterbi Algorithm:
    - Finds the most likely sequence of hidden states given the observed sequence.
      $\delta_t(j) = \max_{i} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(O_t)$
      $\psi_t(j) = \arg\max_{i} [\delta_{t-1}(i) \cdot a_{ij}]$
      
