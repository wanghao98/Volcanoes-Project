#### Data input
pulocation = "/Users/hannahpu/OneDrive - University of Illinois - Urbana/STAT432/Project/"
load(paste0(pulocation, "train_data.RData"))
load(paste0(pulocation, "test_data.RData"))
train.x = train_data
test.x = test_data
train.y = read.csv(paste0(pulocation, "train_labels.csv"), header = TRUE)
test.y = read.csv(paste0(pulocation, "test_labels.csv"), header = TRUE)

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




