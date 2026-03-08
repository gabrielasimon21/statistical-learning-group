# install.packages("ggplot2")
# install.packages("corrplot")
# install.packages("dplyr")
# install.packages("patchwork")
# install.packages("fastDummies")
# install.packages("moments")
# install.packages("Rtsne")
# install.packages("factoextra")
# install.packages("cluster")

library(Rtsne)
library(ggplot2)
library(corrplot)
library(dplyr)
library(patchwork)
library(fastDummies)
library(factoextra)
library(cluster)
library(moments)

set.seed(123)

# URL for the dataset
# https://www.kaggle.com/datasets/vetrirah/customer
# Make sure to have the CSV in the same folder as this script

df1 <- read.csv("/Users/gabrielasimon/Downloads/Test.csv")
df2 <- read.csv("/Users/gabrielasimon/Downloads/Train.csv")

df2 <- df2[-11]

df <- bind_rows(df1, df2)

head(df)

# ID: Unique ID
# Gender: Gender of the customer
# Ever_Married: Marital status of the customer
# Age: Age of the customer
# Graduated: Is the customer a graduate?
# Profession: Profession of the customer
# Work_Experience: Work Experience in years
# Spending_Score: Spending score of the customer
# Family_Size: Number of family members for the customer (including the customer)
# Var_1: Anonymised Category for the customer

# Check how many NAs are present in the dataset

sum(is.na(df))
colSums(is.na(df))

# Fill NAs using 0 for Work_Experience, since customers who did not respond to 
# this question are probably not employed, or have never had work experience
# and 1 for Family_Size, since every family is composed by at least 1 (the 
# customer themself)

df_clean <- df %>%
  mutate(
    Work_Experience = ifelse(is.na(Work_Experience), 0, Work_Experience),
    Family_Size     = ifelse(is.na(Family_Size), 1, Family_Size),
    Ever_Married    = ifelse(Ever_Married == "" | is.na(Ever_Married), "No",      Ever_Married),
    Graduated       = ifelse(Graduated    == "" | is.na(Graduated),    "Unknown", Graduated),
    Gender          = ifelse(Gender       == "" | is.na(Gender),        "Unknown", Gender),
    Profession      = ifelse(Profession   == "" | is.na(Profession),   "Unknown", Profession)
  ) %>%
  select(2:9)

df_clean <- df_clean %>% filter(Profession != "Unknown")

# Remove ID and Var_1 columns
# ID does not provide any information on consumer behaviour
# Var_1 column is an anonymized category that is not explained

# This command should now return 0

sum(is.na(df_clean))
head(df_clean)

# Check for outliers

p1 <- ggplot(df_clean, aes(y = Age))             + geom_boxplot(fill = "#00AFBB") + theme_minimal()
p2 <- ggplot(df_clean, aes(y = Work_Experience)) + geom_boxplot(fill = "#E7B800") + theme_minimal()
p3 <- ggplot(df_clean, aes(y = Family_Size))     + geom_boxplot(fill = "#FC4E07") + theme_minimal()
p1 + p2 + p3

# Some outliers present, but all the values are possible in the real world,
# so all the values will be kept 

# Check skewness and kurtosis of numerical variables

skewness(df_clean$Age);             kurtosis(df_clean$Age)
skewness(df_clean$Family_Size);     kurtosis(df_clean$Family_Size)
skewness(df_clean$Work_Experience); kurtosis(df_clean$Work_Experience)

df_encoded_vars <- df_clean %>%
  mutate(
    # Ordinal encoding for spending to keep the order
    Spending_Score_ord = case_when(
      Spending_Score == "Low"     ~ 1L,
      Spending_Score == "Average" ~ 2L,
      Spending_Score == "High"    ~ 3L
    ),
    # Binary one-hot encoding
    Graduated_bin    = ifelse(Graduated    == "Yes", 1L, 0L),
    Ever_Married_bin = ifelse(Ever_Married == "Yes", 1L, 0L),
    Gender_bin       = ifelse(Gender       == "Male", 1L, 0L)
  )

# After thorough testing, it was concluded that the variable profession did not 
# carry much value for clustering, so it is removed in order to focus only 
# on the demographic data 

df_cluster_vars <- df_encoded_vars %>%
  select(Age, Work_Experience, Family_Size,
         Spending_Score_ord, Graduated_bin, Ever_Married_bin, Gender_bin)


# Check correlations between columns
cor_matrix <- cor(df_cluster_vars)
corrplot(cor_matrix, method = "color", type = "lower",
         tl.col = "black", number.cex = 0.7, tl.cex = 0.8,
         title = "\n\nCorrelation of Clustering Variables")

# Scale the data to find the ideal amount of clusters 
df_scaled <- scale(df_cluster_vars)

# -------

# K MEANS CLUSTERING

# PCA: check how many variables explain 80% of the variance
res.pca <- prcomp(df_scaled, center = TRUE, scale. = FALSE)
cumvar  <- cumsum(res.pca$sdev^2 / sum(res.pca$sdev^2))
n_comp  <- which(cumvar >= 0.80)[1]
cat("Components explaining 80% variance:", n_comp, "\n")

df_pca_scores <- as.data.frame(res.pca$x[, 1:n_comp])

# Variable factor map
fviz_pca_var(res.pca,
             col.var       = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel         = TRUE,
             title         = "Variable Factor Map")

# Elbow Plot
fviz_nbclust(df_pca_scores, kmeans, method = "wss",        nstart = 25) +
  labs(title = "Elbow Plot (PCA space)")

# Silhouette Plot
fviz_nbclust(df_pca_scores, kmeans, method = "silhouette", nstart = 25) +
  labs(title = "Silhouette Plot (PCA space)")

# Adjust the number of centers based on what the elbow/silhouette plots suggest
# In this case: 3

final_clusters <- kmeans(df_pca_scores, centers = 3, nstart = 25)

fviz_cluster(final_clusters, data = df_pca_scores,
             geom         = "point",
             ellipse.type = "convex",
             ggtheme      = theme_minimal(),
             main         = "Customer Segments")

# Attach cluster labels back to the original data to interpret clusters
df_interpreted <- df_clean %>%
  mutate(Cluster = as.factor(final_clusters$cluster))

fviz_pca_biplot(res.pca,
                label       = "var",
                habillage   = df_interpreted$Cluster,
                addEllipses = TRUE,
                col.var     = "black",
                alpha.ind   = 0.3,
                title       = "Variables vs. Clusters")


# Summary table for interpretation

kmeans_summary <- df_interpreted %>%
  group_by(Cluster) %>%
  summarise(
    Count                = n(),
    Avg_Age              = round(mean(Age), 1),
    Median_Work_Exp      = median(Work_Experience),
    Avg_Family_Size      = round(mean(Family_Size), 1),
    Most_Common_Spending = names(which.max(table(Spending_Score))),
    Percent_Graduated    = paste0(round(mean(Graduated    == "Yes") * 100, 1), "%"),
    Most_Common_Gender   = names(which.max(table(Gender))),
    Percent_Married      = paste0(round(mean(Ever_Married == "Yes") * 100, 1), "%"),
    Top_Profession       = names(which.max(table(Profession)))
  )
print(kmeans_summary)

# Run t-SNE for K Means to determine if the clusters are significant or forced
df_tsne_input         <- as.data.frame(df_scaled)
df_tsne_input$cluster <- as.factor(final_clusters$cluster)
df_unique             <- df_tsne_input %>% distinct()

tsne_results <- Rtsne(as.matrix(df_unique[, -ncol(df_unique)]),
                      perplexity       = 30,
                      check_duplicates = FALSE)

tsne_plot_data <- data.frame(
  X       = tsne_results$Y[, 1],
  Y       = tsne_results$Y[, 2],
  Cluster = df_unique$cluster
)

ggplot(tsne_plot_data, aes(x = X, y = Y, color = Cluster)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "t-SNE Visualization of K-Means Clusters")

# Silhouette for K MEANS
d_km   <- dist(df_pca_scores)
sil_km <- silhouette(final_clusters$cluster, d_km)

fviz_silhouette(sil_km) +
  labs(title = "Silhouette Plot: K-Means") +
  theme_minimal()

# -------

# HIERARCHICAL CLUSTERING

# Using Gower distance instead of one-hot encoding
# Gower distance handles categorical data more efficiently by calculating
# the similarity between data points in datasets with mixed data types
df_final_hc <- df_encoded_vars %>%
  mutate(
    Gender          = as.factor(Gender),
    Ever_Married    = as.factor(Ever_Married),
    Graduated       = as.factor(Graduated),
    Spending_Score  = factor(Spending_Score, levels = c("Low", "Average", "High"), ordered = TRUE)
  ) %>%
  select(Age, Work_Experience, Family_Size,
         Spending_Score, Graduated, Ever_Married, Gender)


# Take a sample to test the hierarchical clustering with gower distance, since
# calculating the gower distance is computationally expensive

df_sample_hc <- df_final_hc %>%
  mutate(row_id = row_number()) %>%
  slice_sample(prop = 0.5) %>%      
  select(-row_id)

df_sample_clean <- df_clean[as.integer(rownames(df_sample_hc)), ]

gower_dist <- daisy(df_sample_hc, metric = "gower",
                    type = list(ordratio = 4))  

hc_gower <- hclust(gower_dist, method = "ward.D2")

# Plot the hierarchical clustering dendogram

plot(hc_gower, labels = FALSE, hang = -1,
     main = "Dendrogram", xlab = "Customer Sample", sub = "Method: Ward.D2")
rect.hclust(hc_gower, k = 4, border = "blue")   

hc_labels <- cutree(hc_gower, k = 4)

# Attach cluster labels back to the original data to interpret clusters
df_sample_results <- df_sample_hc %>%
  mutate(
    HC_Cluster = as.factor(hc_labels),
    Profession = df_sample_clean$Profession    
  )

# Summary table for interpretation

hc_summary <- df_sample_results %>%
  group_by(HC_Cluster) %>%
  summarise(
    Count              = n(),
    Avg_Age            = round(mean(Age), 1),
    Median_Work_Exp    = median(Work_Experience),
    Avg_Family_Size    = round(mean(Family_Size), 1),
    Spending           = as.character(names(which.max(table(Spending_Score)))),
    Percent_Graduated  = paste0(round(mean(Graduated    == "Yes") * 100, 1), "%"),
    Most_Common_Gender = as.character(names(which.max(table(Gender)))),
    Percent_Married    = paste0(round(mean(Ever_Married == "Yes") * 100, 1), "%"),
    Top_Profession     = names(which.max(table(Profession)))
  )
print(hc_summary)

# Map visualization of the clusters 
mds_res <- cmdscale(gower_dist, k = 2)

mds_plot_data <- data.frame(
  X       = mds_res[, 1],
  Y       = mds_res[, 2],
  Cluster = as.factor(hc_labels)
)

ggplot(mds_plot_data, aes(x = X, y = Y, color = Cluster)) +
  geom_point(alpha = 0.6, size = 1.5) +
  theme_minimal() +
  labs(title = "2D Map of Hierarchical Clusters")

# Silhouette for hierarchical clustering

sil_hc <- silhouette(hc_labels, gower_dist)
fviz_silhouette(sil_hc) +
  labs(title = "Silhouette Plot: Hierarchical Clusters") +
  theme_minimal()

# ------

# PAM 
# PARTITIONING AROUND MEDOIDS

# Best method for this dataset since it's robust to noise, works with Gower and 
# can handle both numerical and categorical data 

# Find the best k
sil_scores <- sapply(2:8, function(k) {
  fit <- pam(gower_dist, k = k, diss = TRUE)
  mean(silhouette(fit$clustering, gower_dist)[, 3])
})

plot(2:8, sil_scores, type = "b", pch = 19,
     xlab = "Number of Clusters (k)",
     ylab = "Avg Silhouette Width",
     main = "PAM: Silhouette Width by k")
abline(v = which.max(sil_scores) + 1, col = "red", lty = 2)

best_k <- which.max(sil_scores) + 1
cat("Best k for PAM:", best_k, "\n")

# Fit PAM 
set.seed(123)
pam_fit <- pam(gower_dist, k = best_k, diss = TRUE)

# Numeric version for the cluster map (coordinates only)
df_plot <- df_sample_hc %>%
  mutate(across(where(is.factor), as.numeric))

mds_pam <- cmdscale(gower_dist, k = 2)

mds_pam_df <- data.frame(
  X       = mds_pam[, 1],
  Y       = mds_pam[, 2],
  Cluster = as.factor(pam_fit$clustering)
)

ggplot(mds_pam_df, aes(x = X, y = Y, color = Cluster)) +
  geom_point(alpha = 0.5, size = 1.2) +
  theme_minimal() +
  labs(title = paste0("PAM: Customer Segments (k=", best_k, ")"),
       x = "MDS Dimension 1",
       y = "MDS Dimension 2")

df_sample_hc$PAM_Cluster <- as.factor(pam_fit$clustering)

cat("\nMedoid profiles\n")
print(df_sample_hc[pam_fit$id.med, ])

# Summary table for interpretation 

df_sample_hc$Profession <- df_sample_clean$Profession   

pam_summary <- df_sample_hc %>%
  group_by(PAM_Cluster) %>%
  summarise(
    Count              = n(),
    Avg_Age            = round(mean(Age), 1),
    Median_Work_Exp    = median(Work_Experience),
    Avg_Family_Size    = round(mean(Family_Size), 1),
    Spending           = as.character(names(which.max(table(Spending_Score)))),
    Percent_Graduated  = paste0(round(mean(Graduated    == "Yes") * 100, 1), "%"),
    Most_Common_Gender = as.character(names(which.max(table(Gender)))),
    Percent_Married    = paste0(round(mean(Ever_Married == "Yes") * 100, 1), "%"),
    Top_Profession     = names(which.max(table(Profession)))
  )
print(pam_summary)


# Silhouette for PAM 
sil_pam <- silhouette(pam_fit$clustering, gower_dist)
summary(sil_pam)

fviz_silhouette(sil_pam) +
  theme_minimal() +
  labs(title = paste0("Silhouette Plot: PAM (k=", best_k, ")"))

# -------

# INTERPRETATION OF RESULTS

cat("K-MEANS PROFILES\n")
print(kmeans_summary)

cat("HIERARCHICAL CLUSTERING PROFILES\n")
print(hc_summary)

cat("PAM PROFILES\n")
print(pam_summary)