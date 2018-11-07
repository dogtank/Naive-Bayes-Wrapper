#Iterative Feature Selection wrapper for naive bayes classifier
#Written primarily to interrogate the mushroom dataframe
#but could be used for any data frame where the target variable is listed in the first column
#The program will display the error rate as each additional predictor is added
#Additional analysis is done by hand and is included in the assignment report 

#load all relevent libraries
library(naivebayes)
library(MASS)
library(ggplot2)

#load data
Mushrooms <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/aga
ricus-lepiota.data", header=FALSE, sep=",", dec=".", na.strings=c("?"))
#Review data format once loaded
#View(Mushrooms) 

set.seed(0) #set random seed
no_observations <- dim(Mushrooms)[1] # No. observations - used to split into test & train data
unusedFeatureList <- as.list(colnames(Mushrooms)) #create string of column names 
unusedFeatureList  <- unusedFeatureList[-1] #and remove first column, as this is the target 

no_predictors <- length(unusedFeatureList) # No. predictors.
selectedFeatureList <- rep(1:no_predictors,0)
featuresListError <- rep(1:no_predictors,0)
lowestError <- 1


#Outside loop runs for all predictors. 
#Assumes target variable is 1st column, 
#hence runs from 2nd column to (no of predictors + 1)
for (i in 2:(no_predictors + 1)){
  
  lowestError <- 1 #used to record lowest error for this cycle of predictors
  #run through all unused predictors in turn
  for (j in 1:length(unusedFeatureList)){   
        feature <- unlist(unusedFeatureList[j])
        testFeatureList <- c(selectedFeatureList, feature)
        error <- 1
        accuracySum <- 0
        
        #for each unused predictor, check accuracy 10 times  
        for (k in 1:10){
          test_index <- sample(no_observations, size=as.integer(no_observations*0.2), replace=FALSE)
          # 20% data for test
          training_index <- -test_index # Remaining 80% data observations for training
          #first find the best fit for the given feature
          NaiveBayesModel <- naive_bayes(V1 ~ ., data = Mushrooms[training_index,c("V1",testFeatureList)])
          Pred_class <- predict(NaiveBayesModel, newdata = Mushrooms[test_index, ])
          tab <- table(Pred_class, Mushrooms[test_index,"V1"])
          accuracy <- sum(diag(tab))/sum(tab)
          accuracySum  <- accuracySum + accuracy
        }
        #Find mean accuracy and hence mean error
        accuracySum <- accuracySum / 10
        error <- 1 - accuracySum
        #if this error rate is better than the previous best, replace previous best
        #Note the lower the error the better
        if (error < lowestError) {
            lowestError <- error
            bestFeature <- feature
        }
      
  }
  #At this point we have the best new predictor for the selected predictorlist
  #so add to the selected predictor list 
  selectedFeatureList <- c(selectedFeatureList, bestFeature)
  featuresListError <- c(featuresListError,lowestError * 100) #error recorded as % 
  #and remove the selected feature from the unused list in preperation for next iteration
  unusedFeatureList <- unusedFeatureList[unusedFeatureList != bestFeature]
  
  #Display selected features list on screen so user can see the system is working 
  print(selectedFeatureList)
  
}

#all numbers crunched so combine in a dataframe for review and display
FinalResults <- data.frame(selectedFeatureList,featuresListError,c(1:22))
colnames(FinalResults) <- c("FeatureAdded","ErrorPercentage", "Iteration")
print("List of features in order")
print(selectedFeatureList)                           
View(FinalResults)

#and plot for the user - include line showing full fit accuracy from warmup exercise
ggplot(data = FinalResults) + geom_point(mapping = aes(Iteration,ErrorPercentage)) + labs(x = 'No of Predictors', y = 'Error %', title = "Accuracy of naive bayes ", subtitle = "Illustrating bias-variance tradeoff as forward-feature model initially improves & then overfits") + geom_hline(yintercept = .36,linetypec("dashed"))


