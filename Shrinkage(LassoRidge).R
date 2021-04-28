########################### ISTM 660 PROJECT - Team 1 ############################

library(rstudioapi)  # This is a external library of functions
# Getting the path of your current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())
cat("\014")

########################### RIDGE REGRESSION ###########################


library(glmnet)
day_data = read.csv("day.csv")
day_data = day_data[-c(1,2,4,6,11,14,15)]

#Creating matrix for passing to ridge regression
x=model.matrix(day_data$cnt~.,day_data)[,-1] 
y=day_data$cnt

#b. Determine the best value of lambda through cross-validation with the default k=10 number of
#folds and the default values of lambda used by the glmnet() function.

ridge.mod=glmnet(x,y,alpha=0)
dim(coef(ridge.mod))
set.seed(25) # since we call a cross-validation function - need to use set.seed()
cv.out=cv.glmnet(x,y,alpha=0,nfolds=5,type.measure="mse")
plot(cv.out)

bestlam=cv.out$lambda.min ;bestlam
# Therefore, we see that the value of lambda that results in the smallest cross-validation
# error is 364.9
cv.out$cvm
cv.out$cvm[which.min(cv.out$cvm)]
# train MSE : 1638544

#c. Create training and testing subsets in the usual manner, using 75% of the data for training 
#and train a new model using the training subset of data using the best value of lambda you found
#using cross-validation.

set.seed(25)
train=sample(1:nrow(x), nrow(x)*0.75)
test = (1:nrow(x))[-train]
x.train = x[train,]
x.test  = x[-train,]
y.train = y[train]
y.test=y[-train]
set.seed(25)
ridge.mod=glmnet(x.train,y.train,alpha=0)
set.seed(25)
ridge.pred = predict(ridge.mod, s = bestlam, newx = x[-train,]) 

#d. What is the test MSE associated with the best value of lambda?
# Use best lambda to predict test data
mean((ridge.pred - y.test)^2)
# test MSE : 1964011

#e. Does your best value of lambda depend on the number of folds for cross-validation? 
for (i in 3:10) {
  set.seed(1) 
  cv.out=cv.glmnet(x,y,alpha=0,nfolds=i,type.measure="mse")
  plot(cv.out)
  
  bestlam=cv.out$lambda.min ;
  print(bestlam)
  }

#Answer: No, We can see from the above for loop results. The best value of lambda 
#does NOT depend on the K-folds

cv.out$nzero
cv.out$nzero[which.min(cv.out$cvm)]

#f. What are the estimated coefficients for your best model?
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:9,]

########################### Q 2 : LASSO REGRESSION ###########################


# a. Using the same data as in question 1, determine the best value of lambda using the default values for cross-validations and lambda used by the glmnet() function.
set.seed(25) # since we call a cross-validation function - need to use set.seed()
cv.out=cv.glmnet(x,y,alpha=1,type.measure="mse")
bestlam=cv.out$lambda.min;bestlam
#The best value of lambda is 2.38

# b. Plot the results of your cross validation. What information does this provide to you?
plot(cv.out)
# In the plot output, note that the numbers along the top row - all those 17s indicate
# the number of variables in the model for each value of log(lambda) along the horizontal 
# axis. Note that the first dashed vertical line on the left of the plot denotes the results
# associated with the best value for lambda which as you will see below is about 2.13.
# Note, however, the second dashed vertical line to the right of it.  This corresponds to 
# a model with lambda in which the the cross-validated error is within one standard deviation
# of the smallest.  Since differences within one standard deviation can be attributed to
# chance, just as much as a better model, it is often recommended to use this particular 
# model, rather than the minimum, since it will hedge towards a lower variance model (more bias). This value is close to log(lambda)=5.8
#It also tells that as value of lambda increases, the number of variables dropped decreases and MSE increases


# c. What is the test MSE associated with the best value of lambda (use the same training and testing subsets as in question 1)?
#Commented created the subsets to use the same as in ridge
#set.seed(1)
#train=sample(1:nrow(x), nrow(x)*.75)
#test = (1:nrow(x))[-train]
#x.train = x[train,]
#x.test  = x[-train,]
#y.train = y[train]
#y.test=y[-train]

set.seed(25)
lasso_mod = glmnet(x.train,y.train, alpha = 1)
lasso_pred = predict(lasso_mod, s = bestlam, newx = x.test) # Use best lambda to predict test data
mean((lasso_pred - y.test)^2) # Calculate test MSE
#The test MSE Lasso is 1984072

# d. Which model is better: ridge or lasso. Explain why?
#The test MSE Lasso is 1984072 higher to that of Ridge regression. 
out=glmnet(x,y,alpha=1,lambda=bestlam)# Fit lasso model on full dataset
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:9,]
# The next line of code will show all the coefficients
lasso.coef
# The next line of code will show all the non-zero coefficients
lasso.coef[lasso.coef!=0]

cv.out$cvm 
cv.out$cvm[which.min(cv.out$cvm)] #Train MSE is 1823781

cv.out$nzero
cv.out$nzero[which.min(cv.out$cvm)]

#Since the train MSE is lower than test MSE for Lasso, it seems like lasso is a better model.Also, the train MSE for Ridge is a lot higher than of lasso.













