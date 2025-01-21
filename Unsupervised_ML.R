# Unsupervised Learning

# Install and load important packages

## Install the 'packman' package to load the necessary packages together
install.packages("pacman")

## Load important libraries 
pacman::p_load("cluster",
                  "NbClust",
                  "reader",
                  "readxl",
                  "clValid",
                  "factoextra",
                  "tidyverse",
                  "magrittr",
                  "fpc",
                  "ggplot2",
                  "corrplot",
                  "GGally",
                  "clValid")

# import dataset
# information about the variables can be found at: https://archive.ics.uci.edu/dataset/192/breast+tissue
breastissue <- read_excel("/Users/nigus/Desktop/My files/Nigus's File/CV/Advanced epi/BreastTissue.xls", sheet = "Data")
df <- breastissue[,-1]
df <- breastissue[,-2] # label removed


# Descriptive statistics
summary(df)  # check skewness (if mean>median--> right-skewed)

aggregate(df, by=list(breastissue$Class), mean) # help us understand which classes of the dataset are similar and which are different

## box plot 
par(mfrow= c(2,5), bg = "bisque")
for (i in 2:ncol(df)) {
  boxplot(df[,i], main = names(df)[i],
          col = "chocolate", 
          border = "khaki4", 
          notch = T)
} # there is a significant difference between the scales--> standardization
  # The high number of outliers--> the k-medoids algorithm maybe better than k-means

# Correlation analysis 
ggcorr(df[,-1], method = c("everything", "pearson")) # high correlation between pairs of variables--> PCA

## correlation matrix to show Pearson correlation coefficients
cm <- cor(df)
cm

# Principle Component Analysis (PCA)
data_pca <- prcomp(df[,-1], center = TRUE, scale. = TRUE)
summary(data_pca) # PC1 and PC2 explain 80% of the total variance, while PC1 explains 60% of the total variance
                  # combination of these two components largely summarizes the relationships between variables
## Work with the two components
data_pca$rotation[,1:2] # I0, DA and Max IP variables have the highest weights for PC1
                        # PC2 has particularly high weights for the variables PA500, HFS and A/DA
                        # High values of components indicate cases where high values (in a positive direction) of variables co-occur.

## Plot the PCA to see the contributions of variables
fviz_pca_var(data_pca,
             col.var = "contrib", 
             gradient.cols = (("Pastel2")), 
             repel = TRUE     ) # the variables that make up the PC1 component are DA, DR, P
                                # PA50 and HFS make up the PC2 component

## Plot the PCA to see the contributions of observations
fviz_pca_ind(data_pca,
             col.ind = "cos2",  
             gradient.cols = ("Pastel2"), 
             repel = TRUE) # 103 contributes the most to the PC1
                           # 11 the most to the PC2

## we can also see the numerical values for observations' contributions
which.max(data_pca$x[,1]) # index number of the maximum value for PC1 contribution
which.max(data_pca$x[,2]) # index number of the maximum value for PC2 contribution

# Distance matrix using heatmap
dist_euc <- get_dist(df, stand = TRUE)
fviz_dist(dist_euc) # observations that are close to each other are colored red
                    # observations far away are colored blue
                    # some observations are close to each other--> clustering

# Clustering analysis
pcadata <- predict(data_pca)[,1:2] # high correlations 
                                   # number of variables is high compared to the number of observations
                                   # PCA-applied data set will be used in the clustering analysis

## K-means

### Determination of the optimal number of clusters

#### Elbow method
fviz_nbclust(pcadata , kmeans, nstart = 25, iter.max = 200, method = "wss") +
  labs(subtitle = "Elbow Method") # The elbow is between 5–6–7 clusters

#### Average Silhouette Method
fviz_nbclust(pcadata, # data
            kmeans, # clustering algorithm
            method = "silhouette") # The cluster number with maximum silhouette score is 2
                                   # 3 and 4 clusters numbers is also relatively high

#### Gap Statistic
fviz_nbclust(pcadata, # data
             kmeans, # clustering algorithm
             method = "gap")  # the cluster number with maximum gap statistic is 5
                              # 8 clusters number is also relatively high

#### Calinski — Harabasz
fviz_ch <- function(data) {
  ch <- c()
  for (i in 2:10) {
    km <- kmeans(data, i) # perform clustering
    ch[i] <- calinhara(data, # data
                       km$cluster, # cluster assignments
                       cn=max(km$cluster) # total cluster number
    )
  }
  ch <-ch[2:10]
  k <- 2:10
  plot(k, ch,xlab =  "Cluster number k",
       ylab = "Caliński - Harabasz Score",
       main = "Caliński - Harabasz Plot", cex.main=1,
       col = "dodgerblue1", cex = 0.9 ,
       lty=1 , type="o" , lwd=1, pch=4,
       bty = "l",
       las = 1, cex.axis = 0.8, tcl  = -0.2)
  abline(v=which(ch==max(ch)) + 1, lwd=1, col="red", lty="dashed")
}

fviz_ch(df) # the cluster number with maximum gap statistic is 5
            # 6 clusters number is also relatively high

#### Davies — Bouldin
fviz_db <- function(data) {
  k <- c(2:10)
  nb <- NbClust(data, min.nc = 2, max.nc = 10, index = "db", method = "kmeans")
  db <- as.vector(nb$All.index)
  plot(k, db,xlab =  "Cluster number k",
       ylab = "Davies-Bouldin Score",
       main = "Davies-Bouldin Plot", cex.main=1,
       col = "dodgerblue1", cex = 0.9 ,
       lty=1 , type="o" , lwd=1, pch=4,
       bty = "l",
       las = 1, cex.axis = 0.8, tcl  = -0.2)
  abline(v=which(db==min(db)) + 1, lwd=1, col="red", lty="dashed")
}
fviz_db(df) # the cluster number with minimum Davies-Bouldin score is 2

#### Dunn Index
fviz_dunn <- function(data) {
  k <- c(2:10)
  dunnin <- c()
  for (i in 2:10) {
    dunnin[i] <- dunn(distance = dist(data), clusters = kmeans(data, i)$cluster)
  }
  dunnin <- dunnin[2:10]
  plot(k, dunnin, xlab = "Cluster number k",
       ylab = "Dunn Index",
       main = "Dunn Plot", cex.main=1,
       col = "dodgerblue1", cex = 0.9 ,
       lty=1 , type="o" , lwd=1, pch=4,
       bty = "l",
       las = 1, cex.axis = 0.8, tcl = -0.2)
  abline(v=which(dunnin==max(dunnin)) + 1, lwd=1, col="red", lty="dashed")
}
fviz_dunn(df) # the cluster number with maximum Dunn score is 2

# Overall, methods suggested different number of clusters. 2 and 5 are the most suggested numbers.
# However, from the data, the actual cluster number is 6. 
# Thus, data will be clustered for 2 and 6


# K-means for k = 2

set.seed(1993)
k2m_data <- kmeans(pcadata, 2, nstart=25) 
print(k2m_data) # centroid of the first cluster: (-1.190899, 0.3028045)
                # centroid of the second cluster: (3.016945, -0.7671047)
                # clustering vector indicates to which cluster each data sample belongs
                      # the first 62 samples are in the first cluster, & 14 in the 2nd 
                # sum of within cluster sum of squares: a measure of how homogeneous each cluster is (sum of variance within each cluster)
                    # the 2nd cluster is less homogeneous
                # sum of squares within cluster: the two clusters are relatively well separated

## Cluster plot
fviz_cluster(k2m_data, data = pcadata,
             ellipse.type = "convex", 
             star.plot = TRUE, 
             repel = TRUE, 
             ggtheme = theme_minimal()
)  # the second cluster is less homogeneous
   # separation only occurred in the PC1 dimension?

## Cluster Validation for k = 2

### Internal Cluster Validation

#### Silhouette Analysis
k2m_data <- eclust(pcadata, "kmeans", k = 2, nstart = 25, graph = F)
fviz_silhouette(k2m_data, palette = "jco",
                ggtheme = theme_classic()) # average silhouette score of the cluster 1 is higher than cluster 2
                                           # no observations with negative silhouette score--> the observation is clustered wrongly
                                           # Average silhouette score of the clustering is 0.57

#### Dunn Index
km_stats <- cluster.stats(dist(pcadata), k2m_data$cluster)
km_stats$dunn # The Dunn index takes values ranging from zero to maximum
              # The best dunn value is the maximum value
              # The clustering result is very close to zero ( 0.06490181)

#### Connectivity
connectivity(distance = NULL, k2m_data$cluster, Data = pcadata, neighbSize = 20,
             method = "euclidean") # Connectivity takes values from 0 to infinity. 
                                   # It should be as small as possible.

### External Cluster Validation
breastissue %<>% mutate(Class = case_when(
  Class == "car" ~ 1,
  Class == "fad" ~ 2,
  Class == "mas" ~ 3,
  Class == "gla" ~ 4,
  Class == "con" ~ 5,
  Class == "adi" ~ 6
))  # converts class variable to numeric

#### Corrected Rand Index
cluster.stats(d = dist(pcadata),breastissue$Class, 
              k2m_data$cluster)$corrected.rand # Corrected Rand Index takes values between 0 and 1. 
                                               # It needs to be closer to the 1

#### Meila’s Variation of Information
cluster.stats(d = dist(pcadata),breastissue$Class, 
              k2m_data$cluster)$vi  # takes values between 0 and infinity
                                    # It needs to be closer to the 0

#### Accuracy Rate
table(breastissue$Class, k2m_data$cluster) # only 42 observation is clustered correctly?
                                           # accuracy rate of 0.39



# K-means for k = 6

set.seed(1993)
k6m_data <- kmeans(pcadata, 6, nstart=25) 
print(k6m_data)

## Cluster plot
fviz_cluster(k6m_data, data = pcadata,
             ellipse.type = "convex", 
             star.plot = TRUE, 
             repel = TRUE, 
             ggtheme = theme_minimal()
)  # all of the clusters are homogeneous. 
   # It can also be observed that separation occurred both in the PC1 and PC2 dimensions

## Cluster Validation for k = 6

### Internal Cluster Validation

#### Silhouette Analysis
k6m_data <- eclust(pcadata, "kmeans", k = 6, nstart = 25, graph = F)
fviz_silhouette(k6m_data, palette = "jco",
                ggtheme = theme_classic()) # average silhouette score of the cluster 3 is higher than any other cluster
                                           # no observations with negative silhouette score--> clustered wrongly

#### Dunn Index
km_stats <- cluster.stats(dist(pcadata), k6m_data$cluster)
km_stats$dunn # The Dunn index takes values ranging from zero to maximum
              # The best dunn value is the maximum value
              # The clustering result is very close to zero ( 0.04467225)

#### Connectivity
connectivity(distance = NULL, k6m_data$cluster, Data = pcadata, neighbSize = 20,
             method = "euclidean") # Connectivity takes values from 0 to infinity. 
                                   # It should be as small as possible.

### External Cluster Validation

#### Corrected Rand Index
cluster.stats(d = dist(pcadata),breastissue$Class, 
              k6m_data$cluster)$corrected.rand # Corrected Rand Index takes values between 0 and 1. 
                                               # It needs to be closer to the 1

#### Meila’s Variation of Information
cluster.stats(d = dist(pcadata),breastissue$Class, 
              k6m_data$cluster)$vi  # takes values between 0 and infinity
                                    # It needs to be closer to the 0

#### Accuracy Rate
table(breastissue$Class, k6m_data$cluster) # only 56 observation is clustered correctly
                                           # accuracy rate of 0.52
 

# K-medoids

set.seed(1993)
pam_data <- pam(pcadata,6)
print(pam_data) # six medoids identified as a result of the clustering and their location
                # Each medoid is the instance with the smallest average distance from the other instances in its cluster.
                # Objective function: how well the clustering process was performed--smaller value corresponds to a better clustering result
                   # build: the sum of the total distances calculated during the identification of the medoids
                   # swap: the sum of the total distances calculated when swapping medoids

## Cluster plot
fviz_cluster(pam_data, data = pcadata,
             ellipse.type = "convex", 
             star.plot = TRUE, 
             repel = TRUE, 
             ggtheme = theme_minimal()
) # clusters’ homogeneity is different from k-means for k = 6 clustering plot
  # Separation seems to be occurred in both PC1 and PC2 dimensions
  # within cluster sum of squares of cluster 6 is too much

## Cluster Validation

### Internal cluster validation

#### Silhouette Analysis
pam_data <- eclust(pcadata, "pam", k = 6, nstart = 25, graph = F)
fviz_silhouette(pam_data, palette = "jco",
                ggtheme = theme_classic()) # average silhouette score of the cluster 5 is higher
                                           # no observations with negative silhouette score--> clustered wrongly

#### Dunn Index
pm_stats <- cluster.stats(dist(pcadata), pam_data$cluster)
pm_stats$dunn # The Dunn index takes values ranging from zero to maximum
              # The best dunn value is the maximum value
              # The clustering result is very close to zero ( 0.0193406)
              # This calls into question the success of the clustering

#### Connectivity
connectivity(distance = NULL, pam_data$cluster, Data = pcadata, neighbSize = 20,
             method = "euclidean") # Connectivity takes values from 0 to infinity. 
                                   # It should be as small as possible

### External validation

#### Corrected Rand Index
cluster.stats(d = dist(pcadata),breastissue$Class,
              pam_data$cluster)$corrected.rand # Corrected Rand Index takes values between 0 and 1. 
                                               # It needs to be closer to the 1

#### Meila’s Variation of Information
cluster.stats(d = dist(pcadata),breastissue$Class, 
              pam_data$cluster)$vi # akes values between 0 and infinity. 
                                   # It needs to be closer to the 0

#### Accuracy Rate
table(breastissue$Class, pam_data$cluster) # only 59 observation is clustered correctly
                                           # accuracy rate of this clustering result is 0.55


# Hierarchical Clustering

## Ward’s Linkage Method
dist_euc <- dist(pcadata, method="euclidean")
dist_man <- dist(pcadata, method="manhattan")


hc_e <- hclust(d=dist_euc, method="ward.D2")
hc_m <- hclust(d=dist_man, method="ward.D2")

coph_e <- cophenetic(hc_e)
cor(dist_euc,coph_e)

coph_m <- cophenetic(hc_m)
cor(dist_man,coph_m) # high correlation between the cophenetic distance and the original distance between observations
                     # the clustering solution is preserving the structure of the data well--> Manhattan distance

### Dendogram 
groupward6 <- cutree(hc_m, k = 6)
fviz_dend(hc_m, k = 6, 
          cex = 0.5, 
          color_labels_by_k = TRUE, 
          rect = TRUE ) # shows the clustering for each cluster
### Check with cluster plot
fviz_cluster(list(data = pcadata, cluster = groupward6),
             ellipse.type = "convex", 
             repel = TRUE, 
             show.clust.cent = FALSE, ggtheme = theme_minimal()) # clusters’ homogeneity is different from k-medoids for k = 6 clustering plot.
                                                                 # Separation seems to be occurred in both PC1 and PC2 dimensions
                                                                 # It seems that within cluster 6 only has one observation
                                                                 # There is no overlap between clusters

### Cluster validation

#### Internal cluster validation

##### Silhouette Analysis
ward <- eclust(pcadata, "hclust", k = 6, hc_metric = "manhattan",hc_method = "ward.D2", graph = TRUE)
fviz_silhouette(ward, palette = "jco",
                ggtheme = theme_classic()) # average silhouette score of the cluster 5 is higher
                                           # some observations with negative silhouette score-->the observation is clustered wrongly?

##### Dunn Index
warda <- cluster.stats(dist(pcadata), ward$cluster)
warda$dunn # takes values ranging from zero to maximum. 
           # The best dunn value is the maximum value. 
           # The clustering result is very close to zero (0.07493552)

##### Connectivity
connectivity(distance = NULL, ward$cluster, Data = pcadata, neighbSize = 20,
             method = "euclidean") # takes values from 0 to infinity.  
                                   # It should be as small as possible

#### External cluster validation

##### Corrected Rand Index
cluster.stats(d = dist(pcadata),breastissue$Class, 
              ward$cluster)$corrected.rand # takes values between 0 and 1. 
                                           # It needs to be closer to the 1

##### Meila’s Variation of Information
cluster.stats(d = dist(pcadata),breastissue$Class, 
              ward$cluster)$vi # takes values between 0 and infinity. 
                               # It needs to be closer to the 0

##### Accuracy Rate
table(breastissue$Class, ward$cluster) # only 59 observation is clustered correctly
                                       # accuracy rate is 0.55


## Average Linkage Method
hc_e2 <- hclust(d=dist_euc, method="average")
hc_m2 <- hclust(d=dist_man, method="average")


coph_e2 <- cophenetic(hc_e2)
cor(dist_euc,coph_e2)

coph_m2 <- cophenetic(hc_m2)
cor(dist_man,coph_m2) # high correlation between the cophenetic distance and the original distance between observations
                      # clustering solution is preserving the structure of the data well--> Ward method with Manhattan distance
### Dendogram
groupward6 <- cutree(hc_m2, k = 6)
fviz_dend(hc_m2, k = 6, 
          cex = 0.5, 
          color_labels_by_k = TRUE, 
          rect = TRUE )
### check with cluster plot
fviz_cluster(list(data = pcadata, cluster = groupward6),
             ellipse.type = "convex", 
             repel = TRUE, 
             show.clust.cent = FALSE, ggtheme = theme_minimal()) # similar to the Ward’s Linkage method’s clustering
                                                                 # Separation seems to be occurred in both PC1 and PC2 dimensions. 
                                                                 # It seems that cluster 3 and 6 only have one observation. 
                                                                 # There is no overlap between clusters.

### Cluster validation

#### Internal cluster validation

##### Silhouette Analysis
average <- eclust(pcadata, "hclust", k = 6, hc_metric = "manhattan",hc_method = "average", graph = F)
fviz_silhouette(average, palette = "jco",
                ggtheme = theme_classic()) # average silhouette score of the cluster 2 is higher
                                           # some observations with negative silhouette score-->the observation is clustered wrongly

##### Dunn Index
averagea <- cluster.stats(dist(pcadata), average$cluster)
averagea$dunn # takes values ranging from zero to maximum. 
              # The best dunn value is the maximum value. 
              # The clustering result is very close to zero (0.1259977)

##### Connectivity
connectivity(distance = NULL, average$cluster, Data = pcadata, neighbSize = 20,
             method = "euclidean") # Takes values from 0 to infinity. 
                                   # It should be as small as possible.

#### External cluster validation

##### Corrected Rand Index
cluster.stats(d = dist(pcadata),breastissue$Class, 
              average$cluster)$corrected.rand # takes values between 0 and 1. 
                                              # It needs to be closer to the 1

##### Meila’s Variation of Information
cluster.stats(d = dist(pcadata),breastissue$Class, 
              average$cluster)$vi # takes values between 0 and infinity. 
                                  # It needs to be closer to the 0.

##### Accuracy Rate
table(breastissue$Class, average$cluster) # only 48 observation is clustered correctly
                                          # accuracy rate of this clustering result is 0.45


# Density-Based Clustering
library(dbscan)
kNNdistplot(pcadata, k = 10) # to decide epison value
                             # it seems eps is 3, since there is a descent point in the curve

## cluster with some parameters
db <- fpc::dbscan(pcadata, eps = 1,  MinPts = 10) # minimum number of neighbors (MinPts) used for clustering is set to 10 and the epsilon (eps) value is set to 1
print(db) # cluster 0 has 8 noisy points that cannot be included in any clusters--> density of the points are inhomogeneous
          # this clustering is not successful
## cluster plot
fviz_cluster(db, data = pcadata, stand = FALSE,
             ellipse = FALSE, show.clust.cent = FALSE,
             geom = "point",palette = "jco", ggtheme = theme_classic()) # separation occurs only in the PC1 dimension. 
                                                                        # noisy points seems to close the cluster 2

## Cluster validation

### Internal cluster validation

#### Silhouette Analysis
db_stats <- cluster.stats(dist(pcadata), db$cluster)
db_stats[("avg.silwidth")]

#### Dunn Index
db_stats$dunn # takes values ranging from zero to maximum. 
              # The best dunn value is the maximum value. 
              # The clustering result is very close to zero ( 0.06409203)

#### Connectivity
connectivity(distance = NULL, db$cluster, Data = pcadata, neighbSize = 20,
             method = "euclidean") # takes values from 0 to infinity. 
                                   # It should be as small as possible




# Validation of clustering methods 

library(clValid)
clmethods <- c("kmeans","pam","hierarchical") # setting clustering algorithms to compare.
intern <- clValid(pcadata, # dataset
                  nClust = 2:8, # cluster number
                  clMethods = clmethods, # clustering algorithms
                  validation = "internal" # validation criteria
)
summary(intern) # According to the connectivity, it seems that k-means with 2 clusters is the best clustering algorithm
                # Dunn: it seems that hierarchical clustering with the 2 clusters is the best clustering algorithm
                # Silhouette: hierarchical clustering with the 2 clusters is the best clustering algorithm


# Determining the Best Clustering Algorithm Based on Clustering Analysis Results

## Combine cluster validity metrics of each method computed above
clustering_algorithms <- c("2k-means", "6k-means", "k-medoids", "ward", "average")
silhouette_scores <- c(0.5684102, 0.4768399, 0.4273135, 0.50302, 0.5030)
dunn_scores <- c(0.06490181, 0.04467225, 0.0193406, 0.07493552, 0.1259977)
connectivity_res <- c(4.512074, 48.05129, 62.37064, 43.62086, 33.28467)
corr_rand <- c(0.1966444, 0.2818902, 0.2752918, 0.3164783, 0.1676976)
mvi <- c(1.430785, 1.65172, 1.716308, 1.644904, 1.636093)
accuracy_rate <- c(0.39, 0.52, 0.55, 0.55, 0.45)

results <- data.frame(clustering_algorithms, silhouette_scores, dunn_scores, connectivity_res, corr_rand, mvi, accuracy_rate)
results

## check which algorithm performs better for silhouette index
ggplot(results, aes(y = silhouette_scores, x = clustering_algorithms )) +
  geom_bar(stat= "identity", color="lightsalmon4", fill="sienna1") +
  theme_minimal()+
  labs(title =  "Average Silhouette Scores", subtitle = "for each Clustering Algorithm") +
  xlab("Clustering Algorithm") +
  ylab("Average Silhouette Score") # the best clustering algorithm is k-means with two clusters
                                   # However, we know from the dataset that cluster number is 6--> Hierarchical Clustering algorithm with Ward’s Linkage method is the best

## check which algorithm performs better for the Dunn Index
ggplot(results, aes(y = dunn_scores, x = clustering_algorithms )) +
  geom_bar(stat= "identity", color="lightsalmon4", fill="sienna1") +
  theme_minimal()+
  labs(title = "Dunn Indexes", subtitle = "for each Clustering Algorithm") +
  xlab("Clustering Algorithm") +
  ylab("Dunn Index") # the best Dunn Index is the maximum
                     # the best clustering algorithm is Hierarchical Clustering with Average Linkage method

## check which algorithm performs better for the Connectivity
ggplot(results, aes(y = connectivity_res, x = clustering_algorithms )) +
  geom_bar(stat= "identity", color="lightsalmon4", fill="sienna1") +
  theme_minimal()+
  labs(title = "Connectivity Scores", subtitle = "for each Clustering Algorithm") +
  xlab("Clustering Algorithm") +
  ylab("Connectivity") # the best Connectivity is the minimum
                       # the best clustering algorithm is Hierarchical Clustering with Average Linkage method

## check which algorithm performs better for the Corrected Rand Index
ggplot(results, aes(y = corr_rand, x = clustering_algorithms )) +
  geom_bar(stat= "identity", color="lightsalmon4", fill="sienna1") +
  theme_minimal()+
  labs(title = "Corrected Rand Indexes", subtitle = "for each Clustering Algorithm") +
  xlab("Clustering Algorithm") +
  ylab("Corrected Rand Index") # the best Corrected Rand Index is the maximum
                               # the best clustering algorithm is Hierarchical Clustering with Ward’s Linkage method

## check which algorithm performs better for Meila’s Variation of Information
ggplot(results, aes(y = mvi, x = clustering_algorithms )) +
  geom_bar(stat= "identity", color="lightsalmon4", fill="sienna1") +
  theme_minimal()+
  labs(title = "Meila's Variation of Information", subtitle = "for each Clustering Algorithm") +
  xlab("Clustering Algorithm") +
  ylab("MVI") # the best MVI is the minimum
              # the best clustering algorithm is Hierarchical Clustering with Average Linkage method

## check which algorithm performs better for Accuracy Rate
ggplot(results, aes(y = accuracy_rate, x = clustering_algorithms )) +
  geom_bar(stat= "identity", color="lightsalmon4", fill="sienna1") +
  theme_minimal()+
  labs(title = "Accuracy Rates", subtitle = "for each Clustering Algorithm") +
  xlab("Clustering Algorithm") +
  ylab("Accuracy Rate") # the best clustering algorithm is Hierarchical Clustering with Ward’s Linkage method



# conclusion
## All metrics suggested Hierarchical Clustering as the best clustering algorithm. 
## According to the three metrics, Ward’s Linkage method is the best linkage method. 
## The other three metrics suggested Average Linkage method.
## makes sense to act with the Accuracy Rate, which we can directly measure the success of the clustering algorithm


