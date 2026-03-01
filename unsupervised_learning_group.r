library(ggplot2)  
library(corrplot) 
library(dplyr)   
library(patchwork)
library(fastDummies)

# Make sure to have the CSV in the same folder as this script
df <- read.csv("Test.csv")

head(df)

# URL for the dataset
# https://www.kaggle.com/datasets/vetrirah/customer

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

# Fill NAs using the median to prevent skewed numbers for Work_Experience
# and fractions for Family_Size, which could be the case using the mean


df_clean <- df %>%
  mutate(
    Work_Experience = ifelse(is.na(Work_Experience), 0, Work_Experience),
    Family_Size = ifelse(is.na(Family_Size), 1, Family_Size),
    
    Ever_Married = ifelse(Ever_Married == "" | is.na(Ever_Married), "No", Ever_Married),
    Graduated = ifelse(Graduated == "" | is.na(Graduated), "No", Graduated),
    Gender = ifelse(Gender == "" | is.na(Gender), "Unknown", Gender),
    Profession = ifelse(Profession == "" | is.na(Profession), "Unknown", Profession)
  ) %>%
  select(2:9)

# Remove ID and Var_1 columns
# ID does not provide any information on consumer behaviour
# Var_1 column is an anonymized category that is not explained

# This command should now return 0
sum(is.na(df_clean))

head(df_clean)

# Check for outliers
p1 <- ggplot(df_clean, aes(y = Age)) + geom_boxplot(fill="#00AFBB") + theme_minimal()
p2 <- ggplot(df_clean, aes(y = Work_Experience)) + geom_boxplot(fill="#E7B800") + theme_minimal()
p3 <- ggplot(df_clean, aes(y = Family_Size)) + geom_boxplot(fill="#FC4E07") + theme_minimal()

p1 + p2 + p3

# Some outliers present, but all the values are possible in the real world,
# so all the values will be kept 


# Transform non-numeric columns in one hot encoding not to lose information
df_final <- dummy_cols(df_clean, 
                       select_columns = NULL,      
                       split = "_",                
                       remove_selected_columns = TRUE,
                       remove_first_dummy = FALSE)
head(df_final)

# Remove extra dummies
df_final <- df_final[, c(-4, -6, -8, -19, -20)]

head(df_final)

# -----

# Check correlations between columns
cor_matrix <- cor(df_final)
corrplot(cor_matrix, method = "color", type = "lower", 
         tl.col = "black", number.cex = 0.6,  
         tl.cex = 0.7, 
         title = "\n\n Correlation of Customer Attributes")


# Scale the data to find the ideal amount of clusters 
df_scaled <- scale(df_final)

# Elbow Plot
library(factoextra)
fviz_nbclust(df_scaled, kmeans, method = "wss", nstart = 25) +
  labs(title = "Elbow Plot")

# Silhouette Plot
fviz_nbclust(df_scaled, kmeans, method = "silhouette")

# PCA visualization
res.pca <- prcomp(df_scaled)
fviz_pca_ind(res.pca, 
             geom.ind = "point", 
             col.ind = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             title = "PCA Visualization")

# Check for clusters
set.seed(123) 
final_clusters <- kmeans(df_scaled, centers = 3, nstart = 25)

fviz_cluster(final_clusters, data = df_scaled,
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_minimal(),
             main = "Customer Segments")

# Attach cluster IDs the original cleaned data
df_interpreted <- df_clean %>%
  mutate(Cluster = as.factor(final_clusters$cluster))

# Summary table of the averages
cluster_summary <- df_interpreted %>%
  group_by(Cluster) %>%
  summarise(
    Avg_Age = mean(Age),
    Avg_Experience = mean(Work_Experience),
    Avg_Family = mean(Family_Size),
    Common_Profession = names(which.max(table(Profession))),
    Common_Graduated = names(which.max(table(Graduated))),
    Common_Spending = names(which.max(table(Spending_Score))),
    Count = n()
  )

print(cluster_summary)

ggplot(df_interpreted, aes(x = Cluster, fill = Spending_Score)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", title = "Spending Habits by Island")
