# Load necessary libraries
library(tidyverse)
library(caret)
library(corrplot)

# Load the data
df <- read.csv("F:/Student_data.csv")

# Drop Student ID as it holds no predictive value
df <- df %>% select(-Student_ID)

# Inspect structure and summary


View(df)

str(df)

summary(df)

head(df)

# Count the missing values in each column
colSums(is.na(df))



# 1. Find the boundaries of the "box" (Q1 and Q3)
Q1 <- quantile(df$Study_Hours_Per_Day, 0.25)
Q3 <- quantile(df$Study_Hours_Per_Day, 0.75)

# 2. Calculate the Interquartile Range (IQR)
IQR <- Q3 - Q1

# 3. Define the maximum "normal" limit (the top of the upper whisker)
upper_bound <- Q3 + 1.5 * IQR

# 4. Cap the outliers: If a value is above the limit, bring it down to the limit
df$Study_Hours_Per_Day <- ifelse(df$Study_Hours_Per_Day > upper_bound, upper_bound, df$Study_Hours_Per_Day)



# We use ggplot (which loaded when you ran library(tidyverse)) to draw the plot
ggplot(df, aes(y = Study_Hours_Per_Day)) + 
  geom_boxplot(fill = "lightblue", color = "darkblue") +
  theme_minimal() +
  labs(title = "Checking for Outliers: Study Hours Per Day", 
       y = "Study Hours")


# 1. First, tell R clearly that Gender and Major are categories (factors), not just random words
df$Gender <- as.factor(df$Gender)
df$Major <- as.factor(df$Major)

# 2. Define the One-Hot Encoding model
# The "~ ." part is R shorthand for "look at every column in the dataset"
dummy_model <- dummyVars(" ~ .", data = df)

# 3. Apply the transformation and save it as a new dataset called 'df_encoded'
df_encoded <- data.frame(predict(dummy_model, newdata = df))

# 4. Check out your new columns!
head(df_encoded)

# 1. Visualize the distribution of the target variable (Final CGPA)
ggplot(df_encoded, aes(x = Final_CGPA)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Final CGPA", 
       x = "Final CGPA", 
       y = "Number of Students")

# 2. Calculate the correlation matrix for all variables
cor_matrix <- cor(df_encoded)

# 3. Plot the correlation heatmap
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", number.cex = 0.6)


# 1. Set a random seed so your 80/20 split is exactly the same every time you run the script
set.seed(123)

# 2. Create an index that randomly selects 80% of the rows
# We use Final_CGPA to ensure the split is balanced across high and low grades
train_index <- createDataPartition(df_encoded$Final_CGPA, p = 0.8, list = FALSE)

# 3. Create the Training and Testing datasets
train_data <- df_encoded[train_index, ]
test_data  <- df_encoded[-train_index, ]

# 4. Print the sizes to confirm it worked
cat("Training set rows:", nrow(train_data), "\n")
cat("Testing set rows:", nrow(test_data), "\n")


# 1. Train the model using the training data
# The formula "Final_CGPA ~ ." tells R to predict Final_CGPA using ALL other columns
ml_model <- lm(Final_CGPA ~ ., data = train_data)

# 2. View the detailed statistical summary of the trained model
summary(ml_model)


# 1. Give the test data to the model and ask for its predictions
predictions <- predict(ml_model, newdata = test_data)

# 2. Calculate the error rates (caret package calculates this easily)
# RMSE gives higher penalty to large errors
rmse_value <- RMSE(predictions, test_data$Final_CGPA)

# MAE gives the straightforward average error
mae_value <- MAE(predictions, test_data$Final_CGPA)

cat("Root Mean Squared Error (RMSE):", rmse_value, "\n")
cat("Mean Absolute Error (MAE):", mae_value, "\n")

# 3. Create a visual plot of Actual vs. Predicted grades
results <- data.frame(Actual = test_data$Final_CGPA, Predicted = predictions)

ggplot(results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "darkblue", alpha = 0.5) +
  # Add a red dashed line representing "Perfect Predictions"
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  theme_minimal() +
  labs(title = "Model Test: Actual vs. Predicted Final CGPA",
       x = "Actual CGPA (What the student really got)",
       y = "Predicted CGPA (What the model guessed)")