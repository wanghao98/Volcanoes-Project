#### Packages
library(dplyr)
library(randomForest)
library(MASS)
library(bigmemory)
library(data.table)
library(bigalgebra)
library(irlba)
library(glmnet)

#### Data input
load("train_data.RData")
load("test_data.RData")
train.x = train_data
test.x = test_data
train.y = read.csv("train_labels.csv", header = TRUE)
test.y = read.csv("test_labels.csv", header = TRUE)
train = cbind(volcanoe = train.y[,1], train_data)

#### Check dimension
#train image file (no header) : 7000*12100
#train label file (no header) : 7000*4
#test image file (no header) : 2734*12100
#test label file (header) : 2734*4
dim(train.x)
dim(test.x)
dim(train.y)
dim(test.y)

####check missing values
#only y has NAs
colnames(train.x)[colSums(is.na(train.x)) > 0]
colnames(train.y)[colSums(is.na(train.y)) > 0]
colnames(test.x)[colSums(is.na(test.x)) > 0]
colnames(test.y)[colSums(is.na(test.y)) > 0]

#### EDA
##plot
## rows that have volacoes: 1,10,16,30,35,39
train.x1 <- sapply(train.x[1:7000,], as.numeric)
volplot = function(a){
  m=matrix(a, nrow = 110, byrow = TRUE)
  image(m, col = grey((0:255)/255))
}
volplot(train.x1[1,])
volplot(train.x1[10,])
volplot(train.x1[16,])

####Preprocess
# Dimension reduction using SVD.
preprocess.svd = function(train, n.comp){
  train[is.na(train)] = 0
  z = svd(train[, 2:ncol(train)], nu=n.comp, nv=n.comp)
  s = diag(z$d[1:n.comp])
  train[, 2:ncol(train)] = z$u %*% s %*% t(z$v)
  train
}
n.comp = 12
preprocess.svd(train, n.comp)

#####Model

#Lasso
#current result: accuracy 92.7944 (lambda= 0.001882239)
train.x = as.matrix(train.x)
train.y=as.matrix(train.y[,1])
lasmodel = glmnet(x=train.x, y=train.y, alpha = 1, family = "binomial")

pred = predict(lasmodel, newx=as.matrix(test.x), type = "response")
predfunc= function(a){
  ((sum(as.numeric(a > 0.5) == test.y[,1]))/ length(test.y[,1]))*100
}
pred.cal = apply(pred, 2, predfunc)
pred.cal[which.max(pred.cal)+1]
table(as.numeric(pred[,99] > 0.5), test.y[,1])

#Lasso - with tuning
#accuracy: 92.94% (lambda 0.001573354)
train.x = as.matrix(train.x)
train.y=as.matrix(train.y[,1])
cv.out = cv.glmnet(x=train.x, y=train.y, alpha = 1, family = "binomial")
lam.seq = exp(seq(-6.5, -5, length=100))
lasmodel = glmnet(x=train.x, y=train.y, alpha = 1, family = "binomial", lambda = lam.seq)
pred = predict(lasmodel, newx=as.matrix(test.x), type = "response")
predfunc= function(a){
  ((sum(as.numeric(a > 0.5) == test.y[,1]))/ length(test.y[,1]))*100
}
pred.cal = apply(pred, 2, predfunc)
pred.cal[which.max(pred.cal)]
table(as.numeric(pred[,97] > 0.5), test.y[,1])
#Lasso selection-linear regression
mylasso.coef.min = predict(lasmodel, s=cv.out$lambda.min, type="coefficients")
mylasso.coef.1se = predict(lasmodel, s=cv.out$lambda.1se, type="coefficients")
var.sel = row.names(mylasso.coef.1se)[nonzeroCoef(mylasso.coef.1se)[-1]]
tmp.X = X[, colnames(X) %in% var.sel]


#logistic
logmodel = glm(volcanoe~., data=train.svd, family = binomial)

#LDA
ldmodel = lda(train.x, train.y)

#random forest
rfmodel = randomForest(train.x, y=train.y[,1])

#neural network





