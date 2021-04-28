####################### SUPPORT VECTOR CLASSIFICATION LAB ####################################
library(rstudioapi)  # This is a external library of functions

# Getting the path of your current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())
cat("\014")

library(e1071)
library(caret)
library(MLmetrics)


day_data = read.csv("day.csv")
day_data = day_data[-c(1,2,4,6,11,14,15)]

set.seed(25)
trainindex=sample(nrow(day_data),trunc(nrow(day_data)*.75))
train=day_data[trainindex,c(1:9)]  # train with all predictors
test=day_data[-trainindex,c(1:9)]

# Tune a support vector classifier 

set.seed(25)
tune.out=tune(svm,cnt~.,data=train,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)

# Predict using your best model on the test data and check accuracy
predict.y=predict(bestmod,test)

x = 1:length(test$cnt)
plot(x, test$cnt, pch=18, col="red")
lines(x, predict.y, lwd="1", col="blue")

# accuracy check 
mse = MSE(test$cnt, predict.y)
mae = MAE(test$cnt, predict.y)
rmse = RMSE(test$cnt, predict.y)
r2 = R2(test$cnt, predict.y, form = "traditional")

cat(" MAE:", mae, "\n", "MSE:", mse, "\n", 
    "RMSE:", rmse, "\n", "R-squared:", r2)

# Now redo your analysis with a radial kernal using values of 0.1, 0.5, 1 for 
# tuning on cost and gamma

tune.out=tune(svm,cnt~.,data=train,kernel="radial",ranges=list(cost=c(0.1,0.5,0.8),gamma=c(0.1,0.5,0.8)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)

# Predict using your best model on the test data and check accuracy
predict.y=predict(bestmod,test)

x = 1:length(test$cnt)
plot(x, test$cnt, pch=18, col="red")
lines(x, predict.y, lwd="1", col="blue")


# accuracy check 
mse = MSE(test$cnt, predict.y)
mae = MAE(test$cnt, predict.y)
rmse = RMSE(test$cnt, predict.y)
r2 = R2(test$cnt, predict.y, form = "traditional")

cat(" MAE:", mae, "\n", "MSE:", mse, "\n", 
    "RMSE:", rmse, "\n", "R-squared:", r2)

#  Yep... let's go ahead and try the polynomial kernel, try degrees of 2,3, and 4
#  That is, in your list of ranges, use degree=c(2,3,4).  Use the same costs as 
#  you did for the radial kernel

tune.out=tune(svm,cnt~.,data=train,kernel="polynomial",ranges=list(cost=c(0.1,0.5,1),degree=c(2,3,4)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)


# Predict using your best model on the test data and check accuracy
predict.y=predict(bestmod,test)

x = 1:length(test$cnt)
plot(x, test$cnt, pch=18, col="red")
lines(x, predict.y, lwd="1", col="blue")

# accuracy check 
mse = MSE(test$cnt, predict.y)
mae = MAE(test$cnt, predict.y)
rmse = RMSE(test$cnt, predict.y)
r2 = R2(test$cnt, predict.y, form = "traditional")

cat(" MAE:", mae, "\n", "MSE:", mse, "\n", 
    "RMSE:", rmse, "\n", "R-squared:", r2)
