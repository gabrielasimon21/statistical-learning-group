# Load necessary libraries
library(tidyverse)
library(caret)
library(corrplot)
library(randomForest)
library(glmnet)
# Load the data
df <- read.csv("Student_data.csv")

# Drop Student ID as it holds no predictive value
df <- df %>% select(-Student_ID)

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



## Random forest
set.seed(123)

# TRUE to show which variables mattered the most
rf_model <- randomForest(Final_CGPA ~ ., data = train_data, importance = TRUE)
# Get prediction from the RF method
rf_predictions <- predict(rf_model, newdata = test_data)

## Calculating RF metrics to see how well RF is performing
# Root mean squared error
rf_rmse <- RMSE(rf_predictions, test_data$Final_CGPA)
# Mean Absolute error
rf_mae <- MAE(rf_predictions, test_data$Final_CGPA)

# Aggregating data to a single dataframe 
performance_comparison <- data.frame(
  Model = c("Linear Regression (Simple)", "Random Forest (Complex)"),
  RMSE = c(rmse_value, rf_rmse),
  MAE = c(mae_value, rf_mae)
)

# print(performance_comparison)
#                        Model      RMSE       MAE
# 1 Linear Regression (Simple) 0.1616677 0.1310931
# 2    Random Forest (Complex) 0.1426601 0.1150866

# Interpretation: the RMSE is improving in accuracy: roughly 11.8% improvement:
#   from 0.162 to 0.143 -> 0.019. 0.019/0.1617 = 0,1175 (this is the improvement
#   based on the starting point)

#   The MSE is the average mistake that the model makes. The error is tinier ->
#   This means that on average the random forest method guesses the true value
#   with a higher confidence (0.115 on the possible GPA scale of 4) 
#   -> 0.115/4= 2,8% of the total possible GPA scale

# Plot which variables the model found most important
varImpPlot(rf_model, main = "Variable Importance for Final CGPA")

## Lasso model

# Removing the predicted variable in order to perform lasso
x_train <- as.matrix(train_data[, -which(names(train_data) == "Final_CGPA")])
y_train <- train_data$Final_CGPA

x_test <- as.matrix(test_data[, -which(names(test_data) == "Final_CGPA")])
y_test <- test_data$Final_CGPA

grid <- 10^seq(10, -2, length = 100)
lasso_mod <- glmnet(x_train, y_train, alpha = 1, lambda = grid)

# Cross-validation to find the perfect 
set.seed(123)
cv_out <- cv.glmnet(x_train, y_train, alpha = 1)
bestlam <- cv_out$lambda.min

# Using the best lambda to predict the test matrix
lasso_preds <- predict(lasso_mod, s = bestlam, newx = x_test)

lasso_rmse <- sqrt(mean((lasso_preds - y_test)^2))
lasso_mae <- mean(abs(lasso_preds - y_test))

lasso_coef <- predict(lasso_mod, type = "coefficients", s = bestlam)
print(lasso_coef)

# printing the coefficient we see what are the coefficients the lasso considers
# 15 x 1 sparse Matrix of class "dgCMatrix"
# s0
# (Intercept)            -0.58132906
# Gender.Female           .         
# Gender.Male             .         
# Age                     .         
# Major.Business          .         
# Major.Computer.Science  .         
# Major.Economics         .         
# Major.Engineering       .         
# Major.Mathematics       .         
# Major.Psychology        .         
# Attendance_Pct          0.01024213
# Study_Hours_Per_Day     0.04273240
# Previous_GPA            0.90640637
# Sleep_Hours             .         
# Social_Hours_Week      -0.00021973

## Comparing Linear model, RF and Lasso 

model_results <- data.frame(
  Model = c("Linear Regression", "Random Forest", "Lasso Regression"),
  RMSE = c(
    RMSE(predictions, y_test),
    RMSE(rf_predictions, y_test),
    sqrt(mean((lasso_preds - y_test)^2))
  ),
  MAE = c(
    MAE(predictions, y_test),
    MAE(rf_predictions, y_test),
    mean(abs(lasso_preds - y_test))
  )
)

print(model_results)

# Model      RMSE       MAE
# 1 Linear Regression 0.1616677 0.1310931
# 2     Random Forest 0.1426601 0.1150866
# 3  Lasso Regression 0.1626343 0.1317105

# The conclusion are the same as the previous comparison. Seems like the lasso
# is predicting the same way as the linear model

print(bestlam) # is the best lambda for the cross validation

# error curve plot
plot(cv_out)
