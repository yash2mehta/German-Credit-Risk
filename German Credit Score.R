# Prediction Model for German Credit Score data 

#Loading all necessary libraries or packages
library(ggplot2)
library(tidyverse)
library(reshape)
library(dplyr)
library(naniar)
library(rpart)
library(e1071)
library(pROC)
library(gbm)
library(purrr)
library(Hmisc)
library(rlang)
library(cowplot)
library(vcd)
library(DescTools)
library(corrplot)
library(PerformanceAnalytics)
library(factoextra)
library(NbClust)
library(party)
library(randomForest)
library(caret)
library(xgboost)
library(neuralnet)
library(rpart)
library(rpart.plot)
library(mlbench)
library(glmnet)
library(e1071)
library(caTools)
library(class)
library(xgboost)
library(pROC)
library(rcompanion)
library(coin)
library(FactoMineR)

#Changing font size for chart.correlation  by changing the function itself
trace("chart.Correlation", edit=T)

#Reading csv file
cred_raw <- read.csv(file = "German_Credit_data.csv", header = TRUE)


##UNDERSTANDING DATA

#Plotting the 5-number statistical summary of our data
summary(cred_raw)

#Understanding structure of our data
str(cred_raw)

#Understanding dimensions of data (number of rows and number of columns)
dim(cred_raw)

#Obtaining the column headers of our data
names(cred_raw)

##DATA CLEANING

#Checking for NA values (Missing values) by graphical method
vis_miss(cred_raw)

#Confirming the above and checking for NA or NULL values by eye test
is.na(cred_raw)
is.null(cred_raw)

#Creating backup
credit_data <- cred_raw

names(credit_data)

#Splitting our data into different data types
credit_category <- credit_data[c("status","credit_history","purpose","savings","employment_duration","installment_rate","personal_status_sex","other_debtors","present_residence","property","other_installment_plans","housing","number_credits","job","people_liable","telephone","foreign_worker","credit_risk")]
credit_numerical <- credit_data[c("duration","amount","age","credit_risk")]

# DATA PREPARATION

#Creating bloxplot with outliers for individual features

# 1st boxplot: Duration
boxplot_1 <- ggplot(credit_data, aes(x=1, y=duration)) +
  geom_boxplot(fill="red",outlier.color = "blue",size = 1)

boxplot_1 + theme(axis.title.x=element_text(size=15, face="bold"),axis.title.y=element_text(size=15, face="bold"))

# 2nd boxplot: Amount
boxplot_2 <- ggplot(credit_data, aes(x=1, y=amount)) +
  geom_boxplot(fill="red",outlier.color = "blue",size = 1)

boxplot_2 + theme(axis.title.x=element_text(size=15, face="bold"),axis.title.y=element_text(size=15, face="bold"))

# 3rd boxplot: Age
boxplot_3 <- ggplot(credit_data, aes(x=1, y=age)) +
  geom_boxplot(fill="red",outlier.color = "blue",size = 1)

boxplot_3 + theme(axis.title.x=element_text(size=15, face="bold"),axis.title.y=element_text(size=15, face="bold"))


## GRAPHIC EDA (EXPLORATORY DATA ANALYSIS)

#Plotting histogram of each feature

ggplot(gather(credit_data, cols, value), aes(x = value)) + 
  geom_histogram(bins = 20) +
  facet_wrap(.~cols, scales = "free") + 
  labs(title = "Histograms of Features", x = "Feature Value", y = "Frequency") +
  theme_minimal()

#Plotting kernel density plot of each feature
ggplot(gather(credit_data, cols, value), aes(x = value)) + 
  geom_density() + 
  facet_wrap(.~cols, scales = "free") + 
  labs(title = "Histograms of Features", x = "Feature Value", y = "Frequency") +
  theme_minimal()

#Plotting both kernel density and histogram (overlay)
kde <- ggplot(gather(credit_data, cols, value), aes(x = value)) + 
  geom_histogram(aes(y = ..density..), fill="red") + 
  geom_density(col = "#06038D", size = 2) + 
  facet_wrap(.~cols, scales = "free") + 
  labs(x = "Feature Value", y = "Frequency", size = 20) +
  theme_dark() +
  theme(axis.title = element_text(size = 15, face = "bold"), plot.title = element_text(face = "bold",hjust = 0.5))+
  ggtitle("Histograms of Features")

kde




## NON GRAPHIC EDA (EXPLORATORY DATA ANALYSIS)

#Creating correlation matrix for numerical variables
credit_correlation <- select(credit_data, -purpose, -personal_status_sex) #Removing the nominal variables

#Version 1: Pearson correlation matrix can be applied to all variable types
pearson_correlation = cor(credit_correlation, method = c("pearson"))
View(pearson_correlation)
View(pearson_correlation[, c(13, 14, 15, 16,17,18,19)])

#Version 2: Spearman
spearman_correlation = cor(credit_correlation, method = c("spearman"))
View(spearman_correlation)
View(spearman_correlation[, c(12,13, 14, 15, 16,17,18,19)])


#Repeating the above but also generating a table of p-values

#Pearson correlation matrix
correlation.rcorr = rcorr(as.matrix(pearson_correlation))
correlation.rcorr #Printing matrix

correlation.coeff = correlation.rcorr$r #Printing coefficient matrix as dataframe
correlation.p = correlation.rcorr$P #Printing p-value matrix as dataframe

correlation.p #Printing p-value matrix
correlation.rcorr #Printing coefficient matrix

as.data.frame(correlation.p)

View(correlation.p)

View(correlation.p[, c(12,13,14,15,16,17,18,19)])

#P-values for spearman correlation
correlation.spearman.rcorr = rcorr(as.matrix(credit_correlation),type = "spearman")
correlation.spearman.rcorr #Printing matrix

correlation.spearman.coeff = correlation.spearman.rcorr$r #Printing coefficient matrix as dataframe
correlation.spearman.p = correlation.spearman.rcorr$P #Printing p-value matrix as dataframe

as.data.frame(correlation.spearman.p)

View(correlation.spearman.p)
View(correlation.spearman.p[, c(12,13,14,15,16,17,18,19)])

#visualizing correlation matrix
corrplot(pearson_correlation) #Pearson
corrplot(spearman_correlation) #Spearman

#Generating heatmap using our correlation coefficients
palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = pearson_correlation, col = palette, symm = TRUE)


#Following lecture method for creating correlation matrix
chart.Correlation(credit_correlation, histogram = TRUE)

chart.Correlation(credit_correlation, histogram = TRUE, method = "spearman")

#Creating quantile-normal plot for each feature in numerical

gather(numerical[-4], condition, measurement) %>%
  ggplot(aes(sample = measurement)) +
  facet_wrap(~condition, scales = "free") +
  labs(title = "Q-Q plot of numerical variables", x = "Quantities of data", y = "Quantiles") +
  stat_qq() +
  stat_qq_line()

names(credit_data)


#Creating contingency table for categorical variables
contingency_table <- t(sapply(categorical[,-1],    
                       function(x) tapply(x, credit_data$credit_risk, sum)))
# Print contingency table
contingency_table
View(contingency_table)

extra_contingency_table <- addmargins(contingency_table)


#getting marginals from our contingency table
rowSums(contingency_table)

#getting marginals from our contingency table
colSums(contingency_table)

#Getting percents from contingency table
percent_contingency <- prop.table(contingency_table)*100

View(percent_contingency)

#Doing chi-squared test of our contingency table 
chisq.test(contingency_table)


#Doing fisher's exact test of our contingency table 
#Fisher's exact test is an alternative to chi-squared test used mainly when a chi-squared approximation is not satisfactory. 
fisher.test(contingency_table,simulate.p.value = TRUE)

#Doing G-test
GTest(contingency_table)

#Cochran-Mantel-Haenszel Test
#mantelhaen.test(contingency_table)
#cmh(contingency_table, strata = 17, test = c("17x2"))



#Cramer V test
CramerV(contingency_table)

names(credit_data)
pca_data <- credit_data[,-21]

## FEATURE EXTRACTION
pca_traffic <- princomp(pca_data, cor = TRUE, scores = TRUE)
pca_traffic

#Creating a scree plot
screeplot_2 <- fviz_eig(pca_traffic, barfill = "red", linecolor = "#06038D", ggtheme = theme_classic())
screeplot_2 + theme(plot.title = element_text(face = "bold",hjust = 0.5, size = 20),axis.title.x = element_text(face="bold"), axis.title.y = element_text(face="bold")) + geom_point(size = 2)

#PCA for variables
new_pca_credit <- PCA(pca_data, graph = FALSE)

#Total contribution to PC1 and PC2 

screeplot_3 <- fviz_contrib(new_pca_credit, choice = "var", axes = 1:2, top = 10, fill = "#06038D", ggtheme = theme_classic()) +
theme(plot.title = element_text(face = "bold",hjust = 0.5, size = 20),axis.title.x = element_text(face="bold"), axis.title.y = element_text(face="bold")) +
theme(axis.title = element_text(size = 15))

screeplot_3

#Creating a matrix of PCA plots
fviz_pca_var(pca_traffic, col.var = "contrib", repel = TRUE)

pca_graph <- fviz_pca_var(new_pca_credit, col.var = "contrib", labelsize = 3,
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")) + 
  theme(plot.title = element_text(face = "bold",hjust = 0.5, size = 20),axis.title.x = element_text(face="bold"), axis.title.y = element_text(face="bold")) +
  theme(axis.title = element_text(size = 10))

pca_graph

pca_biplot <- fviz_pca_ind(new_pca_credit, col.ind = "contrib", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE) +
  theme(plot.title = element_text(face = "bold",hjust = 0.5, size = 20),axis.title.x = element_text(face="bold"), axis.title.y = element_text(face="bold")) +
  theme(axis.title = element_text(size = 15))

pca_biplot




## MODEL ANALYSIS


#Creating train column 
credit_data[,"train"] <- ifelse(runif(nrow(credit_data))<0.8, 1, 0)

#Creating trainset and test set
trainset <- credit_data[credit_data$train == "1",]
testset <- credit_data[credit_data$train == "0",]

names(trainset)
names(testset)


#Removing train feature
trainset <- trainset[-22]
testset <- testset[-22] #with credit_risk
testdata <- testset[-21] # without credit_risk

#Checking column headers again
names(trainset)
names(testset)
names(testdata)


#RANDOM FOREST MODEL
# Train the model using the training data
forest_credit <- cforest(credit_risk~., data = trainset, control = cforest_unbiased(mtry = 10, ntree = 50))

# Predict the class labels for the test data
predictions <- predict(forest_credit, newdata = testdata, type = "response")

#Attaching probabilistic output
rf_pred <- ifelse(predictions > 0.5, 1, 0)

#Producing another confusion matrix
random_forest_cm <- table(predicted = rf_pred, actual = testset$credit_risk)
random_forest_cm

#Evaluation of random forest model

random_forest_TP <- random_forest_cm[1,1]
random_forest_FP <- random_forest_cm[1,2]
random_forest_TN <- random_forest_cm[2,2]
random_forest_FN <- random_forest_cm[2,1]

#Calculating accuracy of our model
random_forest_accuracy <- (random_forest_TP + random_forest_TN)/(random_forest_TP + random_forest_FP + random_forest_TN + random_forest_FN)

#Calculating precision of our model
random_forest_precision <- (random_forest_TP)/(random_forest_TP + random_forest_FP)

#Calculating recall (sensitivity) of our model
random_forest_recall <- (random_forest_TP)/(random_forest_TP + random_forest_FN)

#Calculating  of our model
random_forest_sensitivity <- (random_forest_TN)/(random_forest_TN+random_forest_FP)

#Calculating F1 score of our model
random_forest_F1_score <- 2 * ((random_forest_precision * random_forest_recall)/(random_forest_precision + random_forest_recall))

#Calculating Matthews Correlation Coefficient of our model
random_forest_MCC <- (random_forest_TP*random_forest_FN - random_forest_FP*random_forest_FN)/(sqrt((random_forest_TP + random_forest_FP)*(random_forest_TN+random_forest_FN)*(random_forest_TN+random_forest_FP)*(random_forest_TN+random_forest_FN)))

#Summary Table of Evaluation
metrics <- c("Accuracy","Precision","Recall","Sensitivity","F1 Score","MCC")
values <- c(random_forest_accuracy,random_forest_precision,random_forest_recall,random_forest_sensitivity,random_forest_F1_score,random_forest_MCC)

df_results <- data.frame(metrics, values)
df_results
View(df_results)

#Finding which features are responsible for greatest amount of variance
ForestVarImp <- varimp(forest_credit)
barplot(ForestVarImp)


# Model SUPPORT VECTOR MACHINE MODEL
svm_trainset <- trainset
svm_trainset$credit_risk <- as.factor(svm_trainset$credit_risk)

#Type of kernels to use: 'linear', 'polynomial', 'radial basis', 'sigmoid')
svm_credit_risk <- svm(credit_risk~., data = svm_trainset, kernel = "polynomial")

summary(svm_credit_risk)

#Fitting prediction model
svm_pred <- predict(svm_credit_risk, newdata = testdata, type = "response")


#Confusion matrix
svm_cm <- table(predicted = svm_pred, actual = testset$credit_risk)
svm_cm

svm_cm_TP <- svm_cm[1,1]
svm_cm_FP <- svm_cm[1,2]
svm_cm_TN <- svm_cm[2,2]
svm_cm_FN <- svm_cm[2,1]

#Calculating accuracy of our model
svm_cm_accuracy <- (svm_cm_TP + svm_cm_TN)/(svm_cm_TP + svm_cm_FP + svm_cm_TN + svm_cm_FN)

#Calculating precision of our model
svm_cm_precision <- (svm_cm_TP)/(svm_cm_TP + svm_cm_FP)

#Calculating recall (sensitivity) of our model
svm_cm_recall <- (svm_cm_TP)/(svm_cm_TP + svm_cm_FN)

#Calculating  of our model
svm_cm_sensitivity <- (svm_cm_TN)/(svm_cm_TN+svm_cm_FP)

#Calculating F1 score of our model
svm_cm_F1_score <- 2 * ((svm_cm_precision * svm_cm_recall)/(svm_cm_precision + svm_cm_recall))

#Calculating Matthews Correlation Coefficient of our model
svm_cm_MCC <- (svm_cm_TP*svm_cm_FN - svm_cm_FP*svm_cm_FN)/(sqrt((svm_cm_TP + svm_cm_FP)*(svm_cm_TN+svm_cm_FN)*(svm_cm_TN+svm_cm_FP)*(svm_cm_TN+svm_cm_FN)))

#Summary Table of Evaluation
metrics <- c("Accuracy","Precision","Recall","Sensitivity","F1 Score","MCC")
values <- c(svm_cm_accuracy,svm_cm_precision,svm_cm_recall,svm_cm_sensitivity,svm_cm_F1_score,svm_cm_MCC)

svm_results <- data.frame(metrics, values)
svm_results

View(svm_results)

par(pty = "s")
roc(testset$credit_risk, svm_pred, plot = TRUE, col = "red", legacy.axes = TRUE, xlab = "False Positive Rate", ylab = "True Positive Rate", lwd = 2, print.auc = TRUE)




# Fitting KNN Model to training dataset


#Creating train column 
credit_data[,"train"] <- ifelse(runif(nrow(credit_data))<0.8, 1, 0)

#Creating trainset and test set
trainset <- credit_data[credit_data$train == "1",]
testset <- credit_data[credit_data$train == "0",]

names(trainset)
names(testset)


#Removing train feature
trainset <- trainset[-22]
testset <- testset[-22] #with credit_risk
testdata <- testset[-21] # without credit_risk

#Checking column headers again
names(trainset)
names(testset)
names(testdata)



names(credit_data)

classifier_knn <- knn(train = trainset,
                      test = testset,
                      cl = trainset$credit_risk,
                      k = 2)
classifier_knn


actual <- testset$credit_risk


# convert the true and predicted labels to factors
actual <- as.factor(actual)
predicted <- as.factor(classifier_knn)

actual
predicted

# make sure the true and predicted labels have the same levels
levels(predicted) <- levels(actual)

# calculate the confusion matrix
knn_cm <- confusionMatrix(actual, predicted)
knn_cm

#Confusion matrix
knn_cm <- table(predicted = predicted, actual = actual)

knn_cm_TP <- knn_cm[1,1]
knn_cm_FP <- knn_cm[1,2]
knn_cm_TN <- knn_cm[2,2]
knn_cm_FN <- knn_cm[2,1]

#Calculating accuracy of our model
knn_cm_accuracy <- (knn_cm_TP + knn_cm_TN)/(knn_cm_TP + knn_cm_FP + knn_cm_TN + knn_cm_FN)

#Calculating precision of our model
knn_cm_precision <- (knn_cm_TP)/(knn_cm_TP + knn_cm_FP)

#Calculating recall (sensitivity) of our model
knn_cm_recall <- (knn_cm_TP)/(knn_cm_TP + knn_cm_FN)

#Calculating  of our model
knn_cm_sensitivity <- (knn_cm_TN)/(knn_cm_TN+knn_cm_FP)

#Calculating F1 score of our model
knn_cm_F1_score <- 2 * ((knn_cm_precision * knn_cm_recall)/(knn_cm_precision + knn_cm_recall))

#Calculating Matthews Correlation Coefficient of our model
knn_cm_MCC <- (knn_cm_TP*knn_cm_FN - knn_cm_FP*knn_cm_FN)/(sqrt((knn_cm_TP + knn_cm_FP)*(knn_cm_TN+knn_cm_FN)*(knn_cm_TN+knn_cm_FP)*(knn_cm_TN+knn_cm_FN)))

#Summary Table of Evaluation
metrics <- c("Accuracy","Precision","Recall","Sensitivity","F1 Score","MCC")
values <- c(knn_cm_accuracy,knn_cm_precision,knn_cm_recall,knn_cm_sensitivity,knn_cm_F1_score,knn_cm_MCC)

knn_results <- data.frame(metrics, values)
knn_results

View(knn_results)

