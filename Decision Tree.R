# Decision Tree Lab

library(rstudioapi)  # This is a external library of functions
# Getting the path of your current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())
cat("\014")
library(tree) # contains functions for generating trees

day_data = read.csv("day.csv")
day_data = day_data[-c(1,2,4,6,11,14,15)]

names(day_data)
summary(day_data)

# set up a training set using 25 as the random number seed
set.seed(25)

# create a training set of data using half of the available bike rental data
train = sample(1:nrow(day_data), nrow(day_data)*0.75)

# We want to predict the one day stock returns (cnt) with a decision
# tree that uses all of the predictions
# To do so, create the tree using the tree command.

tree.bike_rental=tree(cnt~.,data=day_data,subset=train)

# check the output via summary, you should see that only six variables are used
# note that deviance is simply the sum of squared errors for the decision tree
#
summary(tree.bike_rental)

# examine the tree object itself. In this case "tree.bike_rental", R will provide output
# corresponding to each branch of the tree - noting the split criterion, the number
# of observations in that branch, the deviance, and the overall prediction for the branch

tree.bike_rental

# Plot the tree 
par(mfrow=c(1,1))
plot(tree.bike_rental)
text(tree.bike_rental)

# check for opportunities to improve through cost complexity pruning
set.seed(25)
cv.bike_rental=cv.tree(tree.bike_rental)
plot(cv.bike_rental$size,cv.bike_rental$dev,type='b')
cv.bike_rental

# it looks like a tree of size 5 is best

prune.bike_rental=prune.tree(tree.bike_rental,best=5)
plot(prune.bike_rental)
text(prune.bike_rental)

# Make predictions using the test set of data with your best tree

yhat=predict(prune.bike_rental,newdata=day_data[-train,])
bike_rental.test=day_data[-train,"cnt"]
mean((yhat-bike_rental.test)^2)

# Now plot the results to visualize the predictions relative to the observations

plot(yhat,bike_rental.test)
abline(0,1)
#
# Compute the test RMSE, what does it tell you?
#
sqrt(mean((yhat-bike_rental.test)^2))

mean(bike_rental.test)

