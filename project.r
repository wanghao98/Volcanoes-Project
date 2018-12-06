library(data.table)
library(MASS)
library(ggplot2)
library(glmnet)
library(doParallel)
library(foreach)


train_y <- read.csv("train_labels.csv", header = TRUE)
test_y <- read.csv("test_labels.csv", header = TRUE)

load("train_data.RData")
load("test_data.RData")


train.x = as.matrix(train_data)
train.y=as.matrix(train_y[,1])


train_num = as.matrix(train_y[,4])
train_num[train_num == "NaN"] = 0

test_num = as.matrix(test_y[,4])
test_num[test_num == "NaN"] = 0


cl <- makeCluster(3)
registerDoParallel(cl)

pre = Sys.time()
glm_num <- foreach(j = 1:ncol(train.x), .packages = 'lme4', .combine='rbind') %dopar% {
  
  pix = scale(train.x[,j])
  fit.glm <- summary(glm(train_num ~ pix, family = "poisson"))
  fit.glm$coefficients[2,4]
}

Sys.time() - pre

stopCluster(cl)

rownames(glm_num) = colnames(train.x)
colnames(glm_num) = "glm.p value"

save(glm_num, file = "glm_nunm_result.RData")

totalpix = 250
sortedresults = as.matrix(glm_num[order(glm_num),])
usepix = names(sortedresults[1:totalpix,])

top_loc = match(usepix, rownames(glm_num))

sel.pix = train.x[,top_loc]

training = data.frame(train_num,sel.pix)
colnames(training) = c("volcano_num",colnames(sel.pix))

test.x = as.matrix(test_data)
test.pipx = test.x[,top_loc]

glm.num = glm(volcano_num~., data = training, family = "poisson")

log_pred = predict(glm.num, data.frame(test.pipx), type = "response") # 0.003
sqrt(mean((log_pred - test_num)^2))


las_pix= train.x[,las_index]
training_las = data.frame(train_num,las_pix)
colnames(training_las) = c("volcano_num",colnames(las_pix))


test.las = test.x[,las_index]

glm.las.num = glm(volcano_num~., data = training_las, family = "poisson")

glm.las_pred = predict(glm.las.num, data.frame(test.las), type = "response") 
sqrt(mean((glm.las_pred - test_num)^2)) #RMSE = 


glm_result = list(glm.las.num, glm.num)

save(glm_result, file = "glm_num.RData")







mycv = cv.glmnet(x= train.x, y = train_num, family = "poisson",alpha = 1)
lam.seq = exp(seq(-5, -4.5, length=100))
cv.out = cv.glmnet(x=train.x, y=train_num, alpha = 1, family = "poisson", lambda = lam.seq)
las_num = glmnet(x=train.x, y=train_num, alpha = 1, family = "poisson", lambda = cv.out$lambda.min)



save(las_num, file = "las_num.RData")

las_index = which(las_num$beta != 0)

pred = predict(las_num, newx=as.matrix(test_data), type = "response") #RMSE = 0.01411258
pred
sqrt(mean((pred - test_num)^2))


