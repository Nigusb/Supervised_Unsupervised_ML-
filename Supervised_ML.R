# Supervised Learning

# Loading all the required packages
library(tidyverse)
library(dummy)
library(corrplot)
library(smotefamily)
library(e1071)
library(rpart)
library(rpart.plot)
library(class)
library(randomForest)
library(neuralnet)
library(xgboost)
library(MLmetrics)

# Setting the working directory
setwd("/Users/nigus/Desktop/My files/Nigus's File/CV/Advanced epi")
getwd()

# Information about the dataset is avaialble at https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

# Read the diabetesData.csv into a tibble called diabetesData
diabetesData <- read_csv(file = "diabetes.csv",
                         col_types = "nnnnnnnnl",
                         col_names = TRUE)

# Display diabetesData in the console
print(diabetesData)

# Display the structure of diabetesData in the console
str(diabetesData)

# Display the summary of diabetesData in the console
summary(diabetesData)

# Some features in the dataset such as Glucose, BloodPressure, SkinThickness, 
# Insulin, BMI contain 0s. These are incorrect for that feature and 
# indicate missing value for that column.Converting such values to 'NA'
diabetesData['Glucose'][diabetesData['Glucose'] == 0] <- NA 
diabetesData['BloodPressure'][diabetesData['BloodPressure'] == 0] <- NA 
diabetesData['SkinThickness'][diabetesData['SkinThickness'] == 0] <- NA 
diabetesData['Insulin'][diabetesData['Insulin'] == 0] <- NA 
diabetesData['BMI'][diabetesData['BMI'] == 0] <- NA 

# Display the summary of diabetesData in the console
summary(diabetesData)

# Replacing missing values with their mean. (Mean imputation)
diabetesData <- diabetesData %>%
  mutate(Glucose = ifelse(is.na(Glucose), mean(Glucose, na.rm = TRUE), Glucose),
         BloodPressure = ifelse(is.na(BloodPressure), 
                                mean(BloodPressure, na.rm = TRUE), 
                                BloodPressure),
         SkinThickness = ifelse(is.na(SkinThickness), 
                                mean(SkinThickness, na.rm = TRUE), 
                                SkinThickness),
         Insulin = ifelse(is.na(Insulin), mean(Insulin, na.rm = TRUE), Insulin),
         BMI = ifelse(is.na(BMI), mean(BMI, na.rm = TRUE), BMI))


# Creating a function to display boxplots for all variables
displayAllBoxplots <- function(tibbleDataset) {
  tibbleDataset %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_boxplot(mapping = aes(x = value), color = "#0C234B",
                            fill = "#AB0520")  +
    facet_wrap (~ key, scales = "free") +
    theme_minimal()
}

# Diplaying all Boxplots to check for presence of outliers
displayAllBoxplots(diabetesData)

# Interesting Query 1: Calculating Average BMI and Age
diabetesDataAverageBmiAge <- filter(.data = diabetesData,Outcome == TRUE) %>%
  summarize(AverageAge = mean(Age), AverageBMI = mean(BMI))

diabetesDataAverageBmiAge

# Interesting Query 2: Finding average Glucose value for diabetic and
# non-diabetic patient
diabetesDataMeanGlucose <- diabetesData %>%
  group_by(Outcome) %>%
  summarize(meanGlucose = mean(Glucose))

diabetesDataMeanGlucose

# Interesting Query 3: Finding count of each outcome for each pregnancy value
diabetesDataPregnancies <- diabetesData %>%
  select(Pregnancies, Outcome) %>%
  group_by(Pregnancies, Outcome) %>% count()

diabetesDataPregnancies

# Outliers exist in our dataset which we are not removing. However, we are
# maintaining separate tibbles where we store these outliers for future research
# Pregnancies Outliers
outlierMin <- quantile(diabetesData$Pregnancies, 0.25) -
  (IQR(diabetesData$Pregnancies) * 1.5)
outlierMax <- quantile(diabetesData$Pregnancies, 0.75) +
  (IQR(diabetesData$Pregnancies) * 1.5)
print(paste (outlierMin, outlierMax))

pregnanciesOutliers <- diabetesData %>%
  filter(Pregnancies < outlierMin | Pregnancies > outlierMax)

# BloodPressure Outliers
outlierMin <- quantile(diabetesData$BloodPressure, 0.25) -
  (IQR(diabetesData$BloodPressure) * 1.5)
outlierMax <- quantile(diabetesData$BloodPressure, 0.75) +
  (IQR(diabetesData$BloodPressure) * 1.5)
print(paste (outlierMin, outlierMax))

bloodPressureOutliers <- diabetesData %>%
  filter(BloodPressure < outlierMin | BloodPressure > outlierMax)

# BMI Outliers
outlierMin <- quantile(diabetesData$BMI, 0.25) -
  (IQR(diabetesData$BMI) * 1.5)
outlierMax <- quantile(diabetesData$BMI, 0.75) +
  (IQR(diabetesData$BMI) * 1.5)
print(paste (outlierMin, outlierMax))

BMIOutliers <- diabetesData %>%
  filter(BMI < outlierMin | BMI > outlierMax)

# DiabetesPedigreeFunction Outliers
outlierMin <- quantile(diabetesData$DiabetesPedigreeFunction, 0.25) -
  (IQR(diabetesData$DiabetesPedigreeFunction) * 1.5)
outlierMax <- quantile(diabetesData$DiabetesPedigreeFunction, 0.75) +
  (IQR(diabetesData$DiabetesPedigreeFunction) * 1.5)
print(paste (outlierMin, outlierMax))

DiabetesPedigreeFunctionOutliers <- diabetesData %>%
  filter(DiabetesPedigreeFunction < outlierMin | DiabetesPedigreeFunction > outlierMax)

# Insulin Outliers
outlierMin <- quantile(diabetesData$Insulin, 0.25) -
  (IQR(diabetesData$Insulin) * 1.5)
outlierMax <- quantile(diabetesData$Insulin, 0.75) +
  (IQR(diabetesData$Insulin) * 1.5)
print(paste (outlierMin, outlierMax))

InsulinOutliers <- diabetesData %>%
  filter(Insulin < outlierMin | Insulin > outlierMax)

# SkinThickness Outliers
outlierMin <- quantile(diabetesData$SkinThickness, 0.25) -
  (IQR(diabetesData$SkinThickness) * 1.5)
outlierMax <- quantile(diabetesData$SkinThickness, 0.75) +
  (IQR(diabetesData$SkinThickness) * 1.5)
print(paste (outlierMin, outlierMax))

SkinThicknessOutliers <- diabetesData %>%
  filter(SkinThickness < outlierMin | SkinThickness > outlierMax)

# Creating a function to display histograms for all variables
displayAllHistograms <- function(tibbleDataset) {
  tibbleDataset %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_histogram(mapping = aes(x = value, fill = key),
                              color = "black") +
    facet_wrap (~ key, scales = "free") +
    theme_minimal()
}

# Displaying all Histograms
displayAllHistograms(diabetesData)

# Display a correlation matrix of diabetesData rounded to two decimal places
round(cor(diabetesData),2)

# Display a correlation plot using the ＂number＂ method and
# limit output to the bottom left
corrplot(cor(diabetesData),
         method = "number", 
         type = "lower")

# Normalizing data using min-max normalization
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

diabetesData <- diabetesData %>%
  mutate(Pregnancies = normalize(Pregnancies)) %>%
  mutate(Glucose = normalize(Glucose)) %>%
  mutate(BloodPressure = normalize(BloodPressure)) %>%
  mutate(SkinThickness = normalize(SkinThickness)) %>%
  mutate(Insulin = normalize(Insulin)) %>%
  mutate(BMI = normalize(BMI)) %>%
  mutate(DiabetesPedigreeFunction = normalize(DiabetesPedigreeFunction)) %>%
  mutate(Age = normalize(Age))


# Splitting data set in training and testing data set
set.seed(1234)

sampleSet <- sample(nrow(diabetesData),
                    round(nrow(diabetesData) * 0.75),
                    replace = FALSE)

# Put the records from the 75% sample into diabetesDataTraining 
diabetesDataTraining <- diabetesData[sampleSet,]

# Put the remaining 25% of records into diabetesDataTesting
diabetesDataTesting <- diabetesData[-sampleSet,]

# Do we have class imbalance
summary(diabetesDataTraining$Outcome)

# Store class imbalance magnitutde
classImbalanceMagnitude <- 383/193

# SMOTE
diabetesDataTrainingGlmSmoted <- 
  tibble(SMOTE(X = data.frame(diabetesDataTraining),
               target = diabetesDataTraining$Outcome,
               dup_size = 3)$data)

summary(diabetesDataTrainingGlmSmoted)

# Converting Outcome back to logical datatype
diabetesDataTrainingGlmSmoted <- diabetesDataTrainingGlmSmoted %>%
  mutate(Outcome = as.logical(Outcome))

# Get rid of "class" column in tibble
diabetesDataTrainingGlmSmoted <- diabetesDataTrainingGlmSmoted %>%
  select(-class)

# Check on class imbalance on the smoted dataset
summary(diabetesDataTrainingGlmSmoted$Outcome)

-------------------------------------------------------------------------------
  # Logistic Regression Model 
-------------------------------------------------------------------------------
  
diabetesDataGlmModel <- glm(data = diabetesDataTrainingGlmSmoted,
                            family = binomial,
                            formula = Outcome ~ .)

# Display output of the logistic regression model.
summary(diabetesDataGlmModel)

# The significant variables in our dataset are Pregnancies, Glucose, 
# BloodPressure, BMI, DiabetesPedigreeFunction

# Calculating odds ratio for each coefficient. Above 1 represents increase in
# independent variable will increase odds of increasing diabetes
exp(coef(diabetesDataGlmModel)['Pregnancies'])
exp(coef(diabetesDataGlmModel)['Glucose'])
exp(coef(diabetesDataGlmModel)['BloodPressure'])
exp(coef(diabetesDataGlmModel)['SkinThickness'])
exp(coef(diabetesDataGlmModel)['Insulin'])
exp(coef(diabetesDataGlmModel)['BMI'])
exp(coef(diabetesDataGlmModel)['DiabetesPedigreeFunction'])
exp(coef(diabetesDataGlmModel)['Age'])

# Use model to predict outcomes in the testing dataset
diabetesDataGlmPrediction <- predict(diabetesDataGlmModel,
                                     diabetesDataTesting,
                                     type = "response")

# Display diabetesDataPrediction on the console
print(diabetesDataGlmPrediction)

# Treat anything below or equal to 0.5 as 0, anything above 0.5 as 1
diabetesDataGlmPrediction <- 
  ifelse(diabetesDataGlmPrediction >= 0.5, 1, 0)

# Display diabetesDataPrediction on the console
print(diabetesDataGlmPrediction)

# Create confusion matrix 
diabetesDataGlmConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                        diabetesDataGlmPrediction)

# Display confusion matrix 
print(diabetesDataGlmConfusionMatrix)

# Calculate false positive rate 
diabetesDataGlmConfusionMatrix[1, 2] /
  (diabetesDataGlmConfusionMatrix[1, 2] + 
     diabetesDataGlmConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataGlmConfusionMatrix[2, 1] /
  (diabetesDataGlmConfusionMatrix[2, 1] + 
     diabetesDataGlmConfusionMatrix[2, 2])

# Calculate prediction accuracy 
predictiveAccuracyGLM <- sum(diag(diabetesDataGlmConfusionMatrix) / nrow(diabetesDataTesting))

# Calculate F1 score
diabetesDataTesting$Outcome <- as.factor(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "TRUE", 1,0)
F1_ScoreGLM <- F1_Score(y_pred = factor(diabetesDataGlmPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

#  calculate Recall
RecallGLM <- Recall(y_pred = factor(diabetesDataGlmPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate precision 
PrecisionGLM <- Precision(y_pred = factor(diabetesDataGlmPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate sensitivity 
SenGLM <- Sensitivity(y_pred = factor(diabetesDataGlmPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate specificity
SpGLM <- Specificity(y_pred = factor(diabetesDataGlmPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate ROC AUC
aucGLM <- AUC(y_pred = diabetesDataGlmPrediction, y_true = diabetesDataTesting$Outcome)

# Calculate Gini Coefficient
GiniGLM <- Gini(y_pred = diabetesDataGlmPrediction, y_true = diabetesDataTesting$Outcome)



-------------------------------------------------------------------------------
# K - Nearest Neighbors (KNN) Model 
-------------------------------------------------------------------------------
  
# Serparating label and other variables
diabetesDataLabelsKnn <- diabetesData %>%
  select(Outcome)

diabetesDataKnn <- diabetesData %>%
  select(-Outcome)

# Splitting dataset in training and testing dataset
set.seed(1234)

sampleSetKnn <- sample(nrow(diabetesDataKnn),
                       round(nrow(diabetesDataKnn) * 0.75),
                       replace = FALSE)

# Put the records from the 75% sample into diabetesDataTrainingKnn and 
# diabetesDataTrainingLabelsKnn
diabetesDataTrainingKnn <- diabetesDataKnn[sampleSetKnn,]
diabetesDataTrainingLabelsKnn <- diabetesDataLabelsKnn[sampleSetKnn,]

# Put the records from the 25% sample into diabetesDataTrainingKnn and 
# diabetesDataTrainingLabelsKnn
diabetesDataTestingKnn <- diabetesDataKnn[-sampleSetKnn,]
diabetesDataTestingLabelsKnn <- diabetesDataLabelsKnn[-sampleSetKnn,]

# Generate the k-nearest neighbors model 
diabetesDataKnnPrediction <- knn(train = diabetesDataTrainingKnn,
                                 test = diabetesDataTestingKnn,
                                 cl = diabetesDataTrainingLabelsKnn$Outcome,
                                 k = 24)

# Display the predictions from the testing dataset on the console
print(diabetesDataKnnPrediction)

# Display summary of the predictions from the testing dataset
print(summary(diabetesDataKnnPrediction))

# Evaluate the model by forming a confusion matrix
diabetesDataKnnConfusionMatrix <- table(diabetesDataTestingLabelsKnn$Outcome,
                                        diabetesDataKnnPrediction)

# Display the confusion matrix on the console
print(diabetesDataKnnConfusionMatrix)

# Calculate the model predictive accuracy and store into predictiveAccuracy
predictiveAccuracyKnn <- sum(diag(diabetesDataKnnConfusionMatrix)) / 
  nrow(diabetesDataTestingLabelsKnn)

# Display the predictive accuracy on the console
print(predictiveAccuracyKnn)

# Create a matrix of k-values with their predictive accuracy
# Store the matrix into an object called kValueMatrix.
kValueMatrix <- matrix(data = NA,
                       nrow = 0,
                       ncol = 2)

# Assign column names "k value" and "Predictive accuracy" to the kValueMatrix
colnames(kValueMatrix) <- c("k value", "Predictive Accuracy")

# Loop through odd values of k from 1 up to the number of records in the 
# training dataset. With each pass through the loop, store the k-value along 
# with its predictive accuracy
for (kValue in 1:499) {
  
  # Only calculate predictive accuracy if k value is odd
  if(kValue %% 2 != 0) {
    
    # Generate the model
    diabetesDataKnnPrediction <- knn(train = diabetesDataTrainingKnn,
                                     test = diabetesDataTestingKnn,
                                     cl = diabetesDataTrainingLabelsKnn$Outcome,
                                     k = kValue)
    
    # Generate the confusion matrix
    diabetesDataKnnConfusionMatrix <- 
      table(diabetesDataTestingLabelsKnn$Outcome, 
            diabetesDataKnnPrediction)
    
    # Calculate the predictive accuracy
    predictiveAccuracyKnn <- sum(diag(diabetesDataKnnConfusionMatrix)) / 
      nrow(diabetesDataTestingKnn)
    
    # Add a new row to the kValueMatrix
    kValueMatrix <- rbind(kValueMatrix, c(kValue, predictiveAccuracyKnn))
  }
}

# Display the kValueMatrix on the console to determine the best k-value
print(kValueMatrix)

# With k =5, generate the k-nearest neighbors model 
diabetesDataKnnPrediction <- knn(train = diabetesDataTrainingKnn,
                                 test = diabetesDataTestingKnn,
                                 cl = diabetesDataTrainingLabelsKnn$Outcome,
                                 k = 5)

# Display the predictions from the testing dataset on the console
print(diabetesDataKnnPrediction)

# Display summary of the predictions from the testing dataset
print(summary(diabetesDataKnnPrediction))

# Evaluate the model by forming a confusion matrix
diabetesDataKnnConfusionMatrix <- table(diabetesDataTestingLabelsKnn$Outcome,
                                        diabetesDataKnnPrediction)

# Display the confusion matrix on the console
print(diabetesDataKnnConfusionMatrix)

# Calculate false positive rate 
diabetesDataKnnConfusionMatrix[1, 2] /
  (diabetesDataKnnConfusionMatrix[1, 2] + 
     diabetesDataKnnConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataKnnConfusionMatrix[2, 1] /
  (diabetesDataKnnConfusionMatrix[2, 1] + 
     diabetesDataKnnConfusionMatrix[2, 2])


# Calculate the model predictive accuracy and store into predictiveAccuracy
predictiveAccuracyKnn <- sum(diag(diabetesDataKnnConfusionMatrix)) / 
  nrow(diabetesDataTestingLabelsKnn)

# Display the predictive accuracy on the console
print(predictiveAccuracyKnn)

# Calculate F1 score
diabetesDataTesting$Outcome <- as.factor(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "TRUE", 1,0)
diabetesDataKnnPrediction <- ifelse(diabetesDataKnnPrediction == "TRUE", 1,0)
F1_ScoreKNN <- F1_Score(y_pred = factor(diabetesDataKnnPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

#  calculate Recall
RecallKNN <- Recall(y_pred = factor(diabetesDataKnnPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate precision 
PrecisionKNN <- Precision(y_pred = factor(diabetesDataKnnPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate sensitivity 
SenKNN <- Sensitivity(y_pred = factor(diabetesDataKnnPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate specificity
SpKNN <- Specificity(y_pred = factor(diabetesDataKnnPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate ROC AUC
aucKNN <- AUC(y_pred = diabetesDataKnnPrediction, y_true = diabetesDataTesting$Outcome)

# Calculate Gini Coefficient
GiniKNN <- Gini(y_pred = diabetesDataKnnPrediction, y_true = as.numeric(diabetesDataTesting$Outcome))


-------------------------------------------------------------------------------
# Naive Bayes (NB) Model 
-------------------------------------------------------------------------------
  
# Train the naive Bayes model
diabetesDataNbModel <- naiveBayes(formula = Outcome ~ .,
                                  data = diabetesDataTraining,
                                  laplace = 1)

# Build probabilities for each record in the testing dataset
diabetesDataNbProbability <- predict(diabetesDataNbModel,
                                     diabetesDataTesting,
                                     type = "raw")

# Display probabilities on the console
print(diabetesDataNbProbability)

# Predict classes for each record in the testing dataset 
diabetesDataNbPrediction <- predict(diabetesDataNbModel,
                                    diabetesDataTesting,
                                    type = "class")

# Display prediction on the console
print(diabetesDataNbPrediction)

# Evaluate the model by forming a confusion matrix
diabetesDataNbConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                       diabetesDataNbPrediction)

# Display confusion matrix on the console
print(diabetesDataNbConfusionMatrix)

# Calculate false positive rate 
diabetesDataNbConfusionMatrix[1, 2] /
  (diabetesDataNbConfusionMatrix[1, 2] + 
     diabetesDataNbConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataNbConfusionMatrix[2, 1] /
  (diabetesDataNbConfusionMatrix[2, 1] + 
     diabetesDataNbConfusionMatrix[2, 2])

# Calculate the model predictive accuracy 
predictiveAccuracyNb <- sum(diag(diabetesDataNbConfusionMatrix)) / 
  nrow(diabetesDataTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracyNb)

# Calculate F1 score
diabetesDataTesting$Outcome <- as.factor(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "TRUE", 1,0)
diabetesDataNbPrediction <- ifelse(diabetesDataNbPrediction == "TRUE", 1,0)
F1_ScoreNB <- F1_Score(y_pred = factor(diabetesDataNbPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

#  calculate Recall
RecallNB <- Recall(y_pred = factor(diabetesDataNbPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate precision 
PrecisionNB <- Precision(y_pred = factor(diabetesDataNbPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate sensitivity 
SenNB <- Sensitivity(y_pred = factor(diabetesDataNbPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate specificity
SpNB <- Specificity(y_pred = factor(diabetesDataNbPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate ROC AUC
aucNB <- AUC(y_pred = diabetesDataNbPrediction, y_true = diabetesDataTesting$Outcome)

# Calculate Gini Coefficient
GiniNB <- Gini(y_pred = diabetesDataNbPrediction, y_true = as.numeric(diabetesDataTesting$Outcome))


-------------------------------------------------------------------------------
# Decision Tree model 
------------------------------------------------------------------------------
# Train the decision tree model using the training dataset. 
# Note the complexity parameter of 0.01 is the default value
diabetesDecisonTreeModel <- rpart(formula = Outcome ~ .,
                                  method = "class",
                                  cp = 0.01,
                                  data = diabetesDataTraining)

# Display the decision tree plot
rpart.plot(diabetesDecisonTreeModel)


# Predict classes for each record in the testing dataset
diabetesDecisonTreePrediction <- predict(diabetesDecisonTreeModel,
                                         diabetesDataTesting,
                                         type = "class")

# Display the predictions from diabetesPrediction on the console
print(diabetesDecisonTreePrediction)


# Evaluate the model by forming a confusion matrix 
diabetesDecisonTreeConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                            diabetesDecisonTreePrediction)

# Display the Confusion Matrix on the console
print(diabetesDecisonTreeConfusionMatrix)

# Calculate false positive rate 
diabetesDecisonTreeConfusionMatrix[1, 2] /
  (diabetesDecisonTreeConfusionMatrix[1, 2] + 
     diabetesDecisonTreeConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDecisonTreeConfusionMatrix[2, 1] /
  (diabetesDecisonTreeConfusionMatrix[2, 1] + 
     diabetesDecisonTreeConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracyDecisonTree <- sum(diag(diabetesDecisonTreeConfusionMatrix)) /
  nrow(diabetesDataTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracyDecisonTree)

# Calculate F1 score
diabetesDataTesting$Outcome <- as.factor(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "TRUE", 1,0)
diabetesDecisonTreePrediction <- ifelse(diabetesDecisonTreePrediction == "TRUE", 1,0)
F1_ScoreDT <- F1_Score(y_pred = factor(diabetesDecisonTreePrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

#  calculate Recall
RecallDT <- Recall(y_pred = factor(diabetesDecisonTreePrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate precision 
PrecisionDT <- Precision(y_pred = factor(diabetesDecisonTreePrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate sensitivity 
SenDT <- Sensitivity(y_pred = factor(diabetesDecisonTreePrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate specificity
SpDT <- Specificity(y_pred = factor(diabetesDecisonTreePrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate ROC AUC
aucDT <- AUC(y_pred = diabetesDecisonTreePrediction, y_true = diabetesDataTesting$Outcome)

# Calculate Gini Coefficient
GiniDT <- Gini(y_pred = diabetesDecisonTreePrediction, y_true = as.numeric(diabetesDataTesting$Outcome))


-------------------------------------------------------------------------------
# Random Forests (RF) Model 
-------------------------------------------------------------------------------

# Generate RF model to predict diabetes
rf_model <- randomForest(
  factor(Outcome) ~ ., 
  data = diabetesDataTraining, 
  ntree = 500,       # Number of trees in the forest
  mtry = 3,          # Number of variables randomly sampled at each split
  importance = TRUE,
  proximity = TRUE # Measure variable importance
)
print(rf_model)
plot(rf_model)
diabetesDataTraining$Outcome <- factor(diabetesDataTraining$Outcome)
colors <- c("blue", "red")
MDSplot(rf_model, diabetesDataTraining$Outcome, 
        pch = as.numeric(diabetesDataTraining$Outcome), 
        palette = colors)
legend("topright", 
       legend = levels(diabetesDataTraining$Outcome), 
       fill = colors, 
       title = "Outcome", 
       cex = 0.8)

# Make predictions from the RF model
predictions_rf <- predict(rf_model, diabetesDataTesting)
print(predictions_rf)

# Evaluate the model by forming a confusion matrix
diabetesDataRFConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                       predictions_rf)
print(diabetesDataRFConfusionMatrix)

# Calculate false positive rate 
diabetesDataRFConfusionMatrix[1, 2] /
  (diabetesDataRFConfusionMatrix[1, 2] + 
     diabetesDataRFConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataRFConfusionMatrix[2, 1] /
  (diabetesDataRFConfusionMatrix[2, 1] + 
     diabetesDataRFConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracyRF <- sum(diag(diabetesDataRFConfusionMatrix)) /
  nrow(diabetesDataTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracyRF)

# variable importance
importance(rf_model)
varImpPlot(rf_model)

# Hyperparameter tunning 
tune_grid <- expand.grid(mtry = c(2, 3, 4))
control <- trainControl(method = "cv", number = 5)

rf_tuned <- train(
  factor(Outcome) ~ ., 
  data = diabetesDataTraining, 
  method = "rf", 
  trControl = control, 
  tuneGrid = tune_grid
)

print(rf_tuned)

# Retrain the model with best parameters 
rf_modelBest <- randomForest(
  factor(Outcome) ~ ., 
  data = diabetesDataTraining, 
  ntree = 500,       
  mtry = 3,          # best parameter from the tuning 
  importance = TRUE,
  proximity = TRUE 
)
print(rf_modelBest)

# Make predictions from the retrained RF model
predictions_rf_retrained <- predict(rf_modelBest, diabetesDataTesting)
print(predictions_rf_retrained)

# Evaluate the model by forming a confusion matrix
diabetesDataRFRetrainedConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                                predictions_rf_retrained)
print(diabetesDataRFRetrainedConfusionMatrix)

# Calculate false positive rate 
diabetesDataRFRetrainedConfusionMatrix[1, 2] /
  (diabetesDataRFRetrainedConfusionMatrix[1, 2] + 
     diabetesDataRFRetrainedConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataRFRetrainedConfusionMatrix[2, 1] /
  (diabetesDataRFRetrainedConfusionMatrix[2, 1] + 
     diabetesDataRFRetrainedConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracyRF_retrained <- sum(diag(diabetesDataRFRetrainedConfusionMatrix)) /
  nrow(diabetesDataTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracyRF_retrained)

# Calculate F1 score
diabetesDataTesting$Outcome <- as.factor(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "TRUE", 1,0)
predictions_rf_retrained <- ifelse(predictions_rf_retrained == "TRUE", 1,0)
F1_ScoreRF <- F1_Score(y_pred = factor(predictions_rf_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

#  calculate Recall
RecallRF <- Recall(y_pred = factor(predictions_rf_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate precision 
PrecisionRF <- Precision(y_pred = factor(predictions_rf_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate sensitivity 
SenRF <- Sensitivity(y_pred = factor(predictions_rf_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate specificity
SpRF <- Specificity(y_pred = factor(predictions_rf_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate ROC AUC
aucRF <- AUC(y_pred = predictions_rf_retrained, y_true = diabetesDataTesting$Outcome)

# Calculate Gini Coefficient
GiniRF <- Gini(y_pred = predictions_rf_retrained, y_true = as.numeric(diabetesDataTesting$Outcome))

# variable importance
importance(rf_modelBest)
varImpPlot(rf_modelBest)

-------------------------------------------------------------------------------
# Support Vector Machine (SVM) Model
------------------------------------------------------------------------------- 
# Generate the SVM model to predict diabetes
svm_model <- svm(
  Outcome ~ ., 
  data = diabetesDataTraining, 
  type = "C-classification",  # For classification tasks
  kernel = "radial",          # Kernel types: linear, polynomial, radial, sigmoid
  cost = 1,                   # Regularization parameter
  gamma = 0.1                 # Kernel coefficient for radial basis function
)
print(svm_model)

# Make predictions from the SVM 
predictions_svm <- predict(svm_model, diabetesDataTesting)

# Evaluate the model by forming a confusion matrix
diabetesDataSVMConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                        predictions_svm)
print(diabetesDataSVMConfusionMatrix)

# Calculate false positive rate 
diabetesDataSVMConfusionMatrix[1, 2] /
  (diabetesDataSVMConfusionMatrix[1, 2] + 
     diabetesDataSVMConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataSVMConfusionMatrix[2, 1] /
  (diabetesDataSVMConfusionMatrix[2, 1] + 
     diabetesDataSVMConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracySVM <- sum(diag(diabetesDataSVMConfusionMatrix)) /
  nrow(diabetesDataTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracySVM)

# Hyperparameter tuning
tune_resultSVM <- tune(
  svm, 
  factor(Outcome) ~ ., 
  data = diabetesDataTraining,
  ranges = list(cost = c(0.1, 1, 10), gamma = c(0.01, 0.1, 1))
)

# Best model parameters
best_model <- tune_resultSVM$best.model
summary(best_model)

# Retrain the SVM model using the best parameters from tuning
retrained_svm_model <- svm(
  factor(Outcome) ~ ., 
  data = diabetesDataTraining, 
  type = "C-classification",  
  kernel = "radial",         
  cost = 1,                   
  gamma = 0.01                
)

# Summary of the retrained model
summary(retrained_svm_model)

# Make predictions from the retrained model
predictions_svm_retrained <- predict(retrained_svm_model, diabetesDataTesting)

# Evaluate the model by forming a confusion matrix
diabetesDataSVMConfusionMatrix_re <- table(diabetesDataTesting$Outcome,
                                        predictions_svm_retrained)
print(diabetesDataSVMConfusionMatrix_re)

# Calculate false positive rate 
diabetesDataSVMConfusionMatrix_re[1, 2] /
  (diabetesDataSVMConfusionMatrix_re[1, 2] + 
     diabetesDataSVMConfusionMatrix_re[1, 1])

# Calculate false negative rate 
diabetesDataSVMConfusionMatrix_re[2, 1] /
  (diabetesDataSVMConfusionMatrix_re[2, 1] + 
     diabetesDataSVMConfusionMatrix_re[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracySVM_re <- sum(diag(diabetesDataSVMConfusionMatrix_re)) /
  nrow(diabetesDataTesting)
print(predictiveAccuracySVM_re)

# Calculate F1 score
diabetesDataTesting$Outcome <- as.factor(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "TRUE", 1,0)
predictions_svm_retrained <- ifelse(predictions_svm_retrained == "TRUE", 1,0)
F1_ScoreSVM <- F1_Score(y_pred = factor(predictions_svm_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

#  calculate Recall
RecallSVM <- Recall(y_pred = factor(predictions_svm_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate precision 
PrecisionSVM <- Precision(y_pred = factor(predictions_svm_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate sensitivity 
SenSVM <- Sensitivity(y_pred = factor(predictions_svm_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate specificity
SpSVM <- Specificity(y_pred = factor(predictions_svm_retrained), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate ROC AUC
aucSVM <- AUC(y_pred = predictions_svm_retrained, y_true = diabetesDataTesting$Outcome)

# Calculate Gini Coefficient
GiniSVM <- Gini(y_pred = predictions_svm_retrained, y_true = as.numeric(diabetesDataTesting$Outcome))


-------------------------------------------------------------------------------
# Extreme Gradient Boosting (XGBoost) model
-------------------------------------------------------------------------------
# Split features and labels
set.seed(123)
diabetesDataTesting$Outcome <- as.numeric(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "1", 1,0)
train_features <- as.matrix(diabetesDataTraining[, -ncol(diabetesDataTraining)])
train_labels <- diabetesDataTraining$Outcome 

# Create DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = train_features, label = train_labels)

# Set XGBoost parameters
xgb_params <- list(
  objective = "binary:logistic", # For binary classification
  max_depth = 6,
  eta = 0.3,                     # Learning rate
  eval_metric = "logloss"        # Evaluation metric
)

# Train the model
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,                 # Number of boosting rounds
  verbose = 1
)
# Prepare test data
test_features <- as.matrix(diabetesDataTesting[, -ncol(diabetesDataTesting)])
test_labels <- diabetesDataTesting$Outcome

# Make predictions
predictions_XGB <- predict(xgb_model, test_features)

# Convert probabilities to binary predictions
binary_predictions_XGB <- ifelse(predictions_XGB > 0.5, 1, 0)

# Evaluate the model by forming a confusion matrix
diabetesDataXGBConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                        binary_predictions_XGB)
print(diabetesDataXGBConfusionMatrix)

# Calculate false positive rate 
diabetesDataXGBConfusionMatrix[1, 2] /
  (diabetesDataXGBConfusionMatrix[1, 2] + 
     diabetesDataXGBConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataXGBConfusionMatrix[2, 1] /
  (diabetesDataXGBConfusionMatrix[2, 1] + 
     diabetesDataXGBConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracyXGB <- sum(diag(diabetesDataXGBConfusionMatrix)) /
  nrow(diabetesDataTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracyXGB)

# Importance of the the predictors
xgb.importance(feature_names = colnames(train_features), model = xgb_model)
xgb.plot.importance(xgb.importance(model = xgb_model))

# Hyper-parameter tuning 

# Define grid of parameters
xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150),          # Number of trees
  max_depth = c(3, 6, 9),             # Depth of trees
  eta = c(0.01, 0.1, 0.3),            # Learning rate
  gamma = c(0, 1, 5),                 # Minimum loss reduction
  colsample_bytree = c(0.5, 0.7, 1),  # Features per tree
  min_child_weight = c(1, 5, 10),     # Minimum weight for child nodes
  subsample = c(0.5, 0.7, 1)          # Data subsample ratio
)

# Train control
train_control <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # Number of folds
  verboseIter = TRUE       # Show progress
)

# Train model with grid search
set.seed(123)
train_labels <- as.factor(train_labels)
xgb_tuned <- train(
  x = train_features,
  y = train_labels,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid
)

# Best parameters
print(xgb_tuned$bestTune)

# Retrain with best parameters
set.seed(123)
best_params <- list(
  objective = "binary:logistic",
  max_depth = 9,
  eta = 0.01,
  gamma = 1,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 1
)

final_XGBmodel <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = 100
)
print(final_XGBmodel)

# Make predictions

# Convert probabilities to binary predictions
predictions_XGB_best <- predict(final_XGBmodel, test_features)

# Evaluate performance (Confusion Matrix)
binary_predictions_XGB_best <- ifelse(predictions_XGB_best > 0.5, 1, 0)
diabetesDataXGB_best_ConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                              binary_predictions_XGB_best)
print(diabetesDataXGB_best_ConfusionMatrix)

# Calculate false positive rate 
diabetesDataXGB_best_ConfusionMatrix[1, 2] /
  (diabetesDataXGB_best_ConfusionMatrix[1, 2] + 
     diabetesDataXGB_best_ConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataXGB_best_ConfusionMatrix[2, 1] /
  (diabetesDataXGB_best_ConfusionMatrix[2, 1] + 
     diabetesDataXGB_best_ConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracyXGB_best <- sum(diag(diabetesDataXGB_best_ConfusionMatrix)) /
  nrow(diabetesDataTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracyXGB_best)

# Calculate F1 score
diabetesDataTesting$Outcome <- as.factor(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "TRUE", 1,0)
F1_ScoreXGB <- F1_Score(y_pred = factor(binary_predictions_XGB_best), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

#  calculate Recall
RecallXGB <- Recall(y_pred = factor(binary_predictions_XGB_best), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate precision 
PrecisionXGB <- Precision(y_pred = factor(binary_predictions_XGB_best), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate sensitivity 
SenXGB <- Sensitivity(y_pred = factor(binary_predictions_XGB_best), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate specificity
SpXGB <- Specificity(y_pred = factor(binary_predictions_XGB_best), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate ROC AUC
aucXGB <- AUC(y_pred = binary_predictions_XGB_best, y_true = diabetesDataTesting$Outcome)

# Calculate Gini Coefficient
GiniXGB <- Gini(y_pred = binary_predictions_XGB_best, y_true = as.numeric(diabetesDataTesting$Outcome))


-------------------------------------------------------------------------------
# Neural Networks Model 
-------------------------------------------------------------------------------
# Generate the neural network model to predict Diabetes 
set.seed(123)
diabetesDataNeuralNet <- neuralnet(
  formula = Outcome ~.,
  data = diabetesDataTraining,
  hidden = 3,
  act.fct = "logistic",
  linear.output = FALSE) 

# Display the neural network numeric results
print(diabetesDataNeuralNet$result.matrix)

# Visualize the neural network
plot(diabetesDataNeuralNet)

# Use diabetesDataNeuralNet to generate probabilities 
diabetesDataNeuralNetProbability <- neuralnet::compute(diabetesDataNeuralNet,
                                            diabetesDataTesting)

# Display the probabilities from the testing dataset on the console
print(diabetesDataNeuralNetProbability$net.result)

# Convert probability predictions into 0/1 predictions 
diabetesDataNeuralNetPrediction <-
  ifelse(diabetesDataNeuralNetProbability$net.result > 0.5, 1, 0)

# Display the 0/1 predictions on the console
print(diabetesDataNeuralNetPrediction)

# Evaluate the model by forming a confusion matrix
diabetesDataNeuralNetConfusionMatrix <- table(diabetesDataTesting$Outcome,
                                              diabetesDataNeuralNetPrediction)

# Display the confusion matrix on the console
print(diabetesDataNeuralNetConfusionMatrix)

# Calculate false positive rate 
diabetesDataNeuralNetConfusionMatrix[1, 2] /
  (diabetesDataNeuralNetConfusionMatrix[1, 2] + 
     diabetesDataNeuralNetConfusionMatrix[1, 1])

# Calculate false negative rate 
diabetesDataNeuralNetConfusionMatrix[2, 1] /
  (diabetesDataNeuralNetConfusionMatrix[2, 1] + 
     diabetesDataNeuralNetConfusionMatrix[2, 2])

# Calculate the model predictive accuracy
predictiveAccuracyNeuralNet <- sum(diag(diabetesDataNeuralNetConfusionMatrix)) /
  nrow(diabetesDataTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracyNeuralNet)

# Calculate F1 score
diabetesDataTesting$Outcome <- as.factor(diabetesDataTesting$Outcome)
diabetesDataTesting$Outcome <- ifelse(diabetesDataTesting$Outcome == "TRUE", 1,0)
F1_ScoreNN <- F1_Score(y_pred = factor(diabetesDataNeuralNetPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

#  calculate Recall
RecallNN <- Recall(y_pred = factor(diabetesDataNeuralNetPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate precision 
PrecisionNN <- Precision(y_pred = factor(diabetesDataNeuralNetPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate sensitivity 
SenNN <- Sensitivity(y_pred = factor(diabetesDataNeuralNetPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate specificity
SpNN <- Specificity(y_pred = factor(diabetesDataNeuralNetPrediction), y_true = factor(diabetesDataTesting$Outcome), positive = "1")

# Calculate ROC AUC
aucNN <- AUC(y_pred = diabetesDataNeuralNetPrediction, y_true = diabetesDataTesting$Outcome)

# Calculate Gini Coefficient
GiniNN <- Gini(y_pred = diabetesDataNeuralNetPrediction, y_true = as.numeric(diabetesDataTesting$Outcome))


# Compare algorithms computed above using common ML metrics 
model_comparison <- data.frame(
  Model = c("GLM", "KNN", "Naive Bayes", "Decision tree", "Random forests", "SVM", 
            "XGBOOST", "Neural networks"),
  Accuracy = c(0.72, 0.78, 0.73, 0.77, 0.72, 0.72, 0.74, 0.77),
  F1_Score = c(0.71, 0.70, 0.72, 0.72, 0.60, 0.56, 0.60, 0.69),
  Recall = c(0.90, 0.68, 0.56, 0.73, 0.54, 0.46, 0.50, 0.66),
  Precision = c(0.59, 0.74, 0.70, 0.70, 0.67, 0.71, 0.74, 0.72),
  AUC = c(0.75, 0.76, 0.70, 0.76, 0.69, 0.67, 0.69, 0.75),
  Gini = c(0.52, 0.56, 0.46, 0.57, 0.40, 0.39, 0.43, 0.53)
  )
print(model_comparison)
## Plot model accuracy 
ggplot(model_comparison, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# AUC Bar Plot
barplot(model_comparison$AUC, names.arg = model_comparison$Model, col = "lightblue",
        main = "AUC Comparison Across Models", xlab = "Models", ylab = "AUC",
        ylim = c(0, 1))

