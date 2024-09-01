
#####################
#### Simulating Data:
#####################

# Load necessary libraries
set.seed(123)  # For reproducibility

# Number of individuals
n <- 300

# Simulating blood parameters (Assume some parameters are strongly associated with diabetes)
glucose <- c(rnorm(n/2, mean = 90, sd = 10), rnorm(n/2, mean = 160, sd = 20))  # Lower for non-diabetic, higher for diabetic
insulin <- rnorm(n, mean = 15, sd = 5)     # Normally distributed insulin levels
BMI <- c(rnorm(n/2, mean = 22, sd = 3), rnorm(n/2, mean = 30, sd = 5))         # Lower for non-diabetic, higher for diabetic
age <- rnorm(n, mean = 50, sd = 10)        # Normally distributed age
cholesterol <- rnorm(n, mean = 200, sd = 30) # Normally distributed cholesterol levels


# Simulate diabetes status (1 = diabetes, 0 = no diabetes)
diabetes_status <- c(rep(0, n/2), rep(1, n/2))  # First half non-diabetic, second half diabetic


# Combine data into a dataframe
data <- data.frame(glucose, insulin, BMI, age, cholesterol, diabetes_status)

# Display first few rows of the data
head(data)

##################################
#### Diagnosing Diabetes Using PCA
##################################

# Perform PCA on the blood parameters (excluding the diabetes status)
pca_result <- prcomp(data[, 1:5], scale. = TRUE)  # Scaling the data

# Display summary of PCA results
summary(pca_result)

# Plot the PCA results
library(ggplot2)
pca_data <- as.data.frame(pca_result$x)
pca_data$diabetes_status <- as.factor(data$diabetes_status)

# Plotting the first two principal components
ggplot(pca_data, aes(x = PC1, y = PC2, color = diabetes_status)) +
  geom_point(size = 2) +
  labs(title = "PCA of Blood Parameters for Diabetes Diagnosis",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

# The PCA plot we've generated shows a clear separation between the diabetic and non-diabetic groups along the first principal component (PC1). 
# This separation suggests that PC1 captures most of the variation related to diabetes status.
# To determine which blood parameters are strongly related to diabetes, you should examine the loadings of each parameter on the principal components.
# The loadings indicate how much each original variable (blood parameter) contributes to each principal component.

# Extract the loadings (contributions of each variable to the principal components)
loadings <- pca_result$rotation

# Display the loadings for the first two principal components
loadings[, 1:2]

# Based on the loadings we see below, here's how you can interpret which blood parameters are strongly related to diabetes:
##                    PC1         PC2
## glucose     0.69983720 -0.08780282
## insulin     0.07479256  0.57313484
## BMI         0.70389056 -0.05491851
## age         0.03481210  0.73055140
## cholesterol 0.08921981  0.35649085

## Principal Component 1 (PC1):
## Glucose (0.700) and BMI (0.704) have high positive loadings on PC1.
## This suggests that both glucose and BMI are the most influential parameters in distinguishing between diabetic and non-diabetic individuals.
## Since PC1 separates the two groups, these parameters are likely to be strongly related to diabetes status.

## Principal Component 2 (PC2):
## Age (0.731) and Insulin (0.573) have high positive loadings on PC2.
## PC2 captures some variation that is not directly related to the primary separation of diabetes status but might be linked to other underlying factors or patterns within the data.

## Summary:
## Glucose and BMI are the primary contributors to the separation along PC1, indicating that these parameters are the most strongly related to diabetes in your dataset.
## Age and Insulin contribute to PC2, which might indicate their relevance in explaining other variations in the data that are not directly related to diabetes diagnosis.
## Cholesterol has relatively low loadings on both PC1 and PC2, suggesting it plays a lesser role in distinguishing diabetes status in this analysis.



#################################
### Logistic regression analysis
#################################

# Assuming your data is in a dataframe called `data` with columns for each parameter
# and a binary outcome `diabetes_status` (0 = non-diabetic, 1 = diabetic)

# Fit the logistic regression model
logistic_model <- glm(diabetes_status ~ glucose + insulin + BMI + age + cholesterol, 
                      data = data, 
                      family = binomial)

# Summarize the model to see the p-values
summary(logistic_model)

## There are warnings messages as follow: 
## Warning messages:
## 1: glm.fit: algorithm did not converge 
## 2: glm.fit: fitted probabilities numerically 0 or 1 occurred 
## The warnings indicate that the logistic regression model is having trouble fitting the data. 
## This can happen when the predictors perfectly separate the classes or when the data has extreme values.
## Here’s how you can address these issues:

### 1. Check for Multicollinearity:
	# High multicollinearity among predictors can cause instability in the model.
	# Use the Variance Inflation Factor (VIF) to check for multicollinearity.
	# If VIF values are high (typically above 5 or 10), consider removing or combining highly correlated predictors.

# Install and load the car package if not already installed
install.packages("car")
library(car)

# Calculate VIF
vif(logistic_model)

##     glucose     insulin         BMI         age cholesterol 
##    1.188719    3.856907    2.949668    5.703274    1.168852 

### 2. Regularization:
	# If multicollinearity is not the issue, consider using penalized regression methods like Ridge 
	# or Lasso regression, which can handle cases where predictors are highly correlated or when the model is prone to overfitting.

# For Lasso
library(glmnet)

# Prepare the data
x <- model.matrix(diabetes_status ~ glucose + insulin + BMI + age + cholesterol, data)[, -1]
y <- data$diabetes_status

# Fit the Lasso model with cross-validation
cv_lasso <- cv.glmnet(x, y, family = "binomial", alpha = 1)  # Lasso, alpha = 1

# Optimal lambda from cross-validation
best_lambda_lasso <- cv_lasso$lambda.min

# Fit the final Lasso model using the best lambda
final_lasso_model <- glmnet(x, y, family = "binomial", alpha = 1, lambda = best_lambda_lasso)

# Extract coefficients
lasso_coefficients <- coef(final_lasso_model)

# Display coefficients
print(lasso_coefficients)

## 6 x 1 sparse Matrix of class "dgCMatrix"
                      s0
## (Intercept) -52.67418755
## glucose       0.33672902
## insulin       0.02552832
## BMI           0.30090197
## age           .         
## cholesterol   0.01718897

## Summary of Findings:
## Significant Predictors: Glucose and BMI have the largest coefficients and are included in the final model, indicating they are the most important predictors of diabetes in this Lasso model.
## Insulin and Cholesterol: Both have non-zero coefficients but are less influential compared to glucose and BMI.
## Age: Is excluded from the model by Lasso regularization, suggesting it is not important for predicting diabetes in this context.


# For Ridge Regression

# Fit the Ridge model with cross-validation
cv_ridge <- cv.glmnet(x, y, family = "binomial", alpha = 0)  # Ridge, alpha = 0

# Optimal lambda for Ridge
best_lambda_ridge <- cv_ridge$lambda.min

# Fit the final Ridge model using the best lambda
final_ridge_model <- glmnet(x, y, family = "binomial", alpha = 0, lambda = best_lambda_ridge)

# Extract coefficients
ridge_coefficients <- coef(final_ridge_model)

# Display coefficients
print(ridge_coefficients)

## > print(ridge_coefficients)
## 6 x 1 sparse Matrix of class "dgCMatrix"
                       s0
## (Intercept) -10.534199508
## glucose       0.047065065
## insulin       0.019071472
## BMI           0.154171331
## age          -0.006581047
## cholesterol   0.003793110

## Summary:
## BMI and glucose are the most influential parameters in this Ridge model, with the largest positive coefficients.
## Insulin and cholesterol have smaller coefficients but still contribute positively to diabetes risk.
## Age has a negative coefficient in this model, suggesting it may not be a strong predictor or might be counterintuitive given the dataset and regularization applied.





##################
### Random forest
##################

# Install and load necessary libraries
if (!require(randomForest)) install.packages("randomForest", dependencies=TRUE)
library(randomForest)

# Load the simulated data (assuming you have it as 'data' from previous steps)
# data <- ... # Use the previously simulated data

# Convert diabetes_status to a factor for classification
data$diabetes_status <- as.factor(data$diabetes_status)

# Split the data into training and testing sets (80% train, 20% test)
set.seed(123)  # For reproducibility
train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Build Random Forest model
rf_model <- randomForest(diabetes_status ~ ., data = train_data, importance = TRUE, ntree = 100)

# Print the model summary
print(rf_model)

# Predict on the test data
predictions <- predict(rf_model, newdata = test_data)

# Confusion matrix to evaluate the model
confusion_matrix <- table(test_data$diabetes_status, predictions)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

# Plot feature importance
importance_plot <- importance(rf_model)
print(importance_plot)

# Optional: Plot feature importance using ggplot2
## The plot shows the importance of each blood parameter in the Random Forest model, 
## helping you identify which parameters are most relevant for predicting diabetes.

library(ggplot2)
importance_df <- as.data.frame(importance_plot)
importance_df$Feature <- rownames(importance_df)
ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance from Random Forest",
       x = "Feature",
       y = "Mean Decrease in Gini Index") +
  theme_minimal()

## The plot shows the Mean Decrease in Gini Index (or Mean Decrease in Accuracy) for each feature. 
## This metric represents the importance of each feature in the model.
## A higher value indicates a more important feature. 
## Features with higher importance are more useful in making accurate predictions.

## In the figure, the length of each bar in the plot corresponds to the Mean Decrease in Gini Index.
## Longer bars indicate higher importance.

## Features are ordered based on their importance. 
## Features at the top of the list (with longer bars) are more influential in the Random Forest model.

