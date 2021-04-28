library(rstudioapi)  
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())
cat("\014")

day_data = read.csv("day.csv")
day_data = day_data[-c(1,2,4,6,11,14,15)]


#Random Forest
library(randomForest)

# set up training data set
set.seed(25)
train= sample(1:nrow(day_data),nrow(day_data)*0.75)


# training the model
rfModel <- randomForest(cnt ~.,
                        data = day_data[train,])
rfModel
plot(rfModel)


#Tuning ntree value
which.min(rfModel$mse)  #360
sqrt(rfModel$mse[which.min(rfModel$mse)]) #1209.364
#Tuning mtry value
x=day_data[,1:8]
y=day_data[,9]
bestmtry <- tuneRF(x, y, stepFactor=1.5, improve=0.05, ntree=500)

print(bestmtry)

#After Tuning
rfModel <- randomForest(cnt ~.,data = day_data[train,],mtry=3,ntree=360,importance=TRUE,)
rfModel

#Evaluate variable importance
importance(rfModel)
varImpPlot(rfModel)

# Checking the predictions relative to observed for the test set
rfPred=predict(rfModel,newdata=day_data[-train,])

# plotting the results of actual versus predicted
bikerental.test=day_data[-train,"cnt"]
plot(rfPred, bikerental.test, xlab= "Predicted # of users", ylab = "Observed # of users")
abline(0,1)

#MSE
mse = (mean((rfPred-bikerental.test)^2))
#RMSE
rmse=sqrt(mse)

 
   
  




