## Analysis of K-Means Clustering and Supervised Learning on Mall Customers and Breast Cancer

This project applies both unsupervised and supervised machine learning techniques to analyze two datasets: a mall customer segmentation dataset and a breast cancer dataset. The notebook demonstrates how to perform K-Means clustering with Manhattan distance and how to apply various classification algorithms to validate the clustering results and analyze the breast cancer dataset.

### Project Highlights:

1. **K-Means Clustering on Mall Customers Dataset:**
   - **Objective:** Segment customers based on their annual income and spending score using K-Means clustering.
   - **Distance Metric:** Manhattan distance is utilized for clustering, ensuring robust and meaningful customer segments.
   
   ![K-Means Clustering Visualization](path_to_kmeans_clustering_plot.png)

2. **Classification Algorithms:**
   - **Logistic Regression, Decision Tree, Random Forest, and Naive Bayes:**
     - These algorithms are applied to classify the customer segments identified by the K-Means algorithm.
   - **Performance Comparison:**
     - The models are evaluated based on accuracy, providing insights into the effectiveness of the clustering and the distinctness of the customer segments.

   ![Classifier Accuracy Comparison](path_to_classifier_accuracy_plot.png)

3. **Breast Cancer Dataset Analysis:**
   - **Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA):**
     - Both PCA and LDA are applied to the breast cancer dataset to reduce dimensionality and visualize the data in a lower-dimensional space.
   - **Component Analysis:**
     - The components that explain the maximum variance are identified, highlighting the most significant features for classification.

   ![PCA and LDA Visualization](path_to_pca_lda_plot.png)

### Detailed Workflow:

1. **K-Means Clustering on Mall Customers:**
   - **Clustering Visualization:** The dataset is segmented into clusters using K-Means with Manhattan distance, and the clusters are visualized in a scatter plot to reveal spending behavior patterns among different customer segments.

2. **Supervised Learning on Clusters:**
   - **Algorithm Evaluation:** Logistic Regression and Naive Bayes achieved the highest accuracy (97.5%), while Decision Tree and Random Forest classifiers also performed well (92.5%). These results validate the meaningfulness of the customer clusters.
   - **Implications:** The consistent performance across different algorithms suggests that the customer segments are distinct and well-separated, providing valuable insights for targeted marketing strategies.

3. **Breast Cancer Dataset Analysis:**
   - **PCA and LDA Comparison:** PCA and LDA are applied to the breast cancer dataset to identify the most informative components. The first principal component in PCA explains 44.27% of the variance, while LDA provides a single linear discriminant for classification.
   - **Visualization:** The reduced-dimensional plots for PCA and LDA reveal the underlying structure of the data, aiding in the understanding of breast cancer classifications.

### How to Use:

1. **Clone the Repository:**
   - Clone this repository to your local machine using `git clone`.
   
2. **Install Dependencies:**
   - Install the required Python packages with `pip install -r requirements.txt`.

3. **Run the Notebook:**
   - Open the notebook in Jupyter and execute the cells in sequence to reproduce the analysis and generate the plots.

### Visual Examples:

- **K-Means Clustering:** 
  ![K-Means Clustering Visualization](path_to_kmeans_clustering_plot.png)
  
- **Classifier Accuracy Comparison:** 
  ![Classifier Accuracy Comparison](path_to_classifier_accuracy_plot.png)
  
- **PCA and LDA Visualization:** 
  ![PCA and LDA Visualization](path_to_pca_lda_plot.png)

### Conclusion:

This project effectively demonstrates the application of both K-Means clustering and supervised learning techniques. The results validate the use of these methods for customer segmentation and breast cancer classification, providing valuable insights that can drive business decisions and medical diagnoses.
