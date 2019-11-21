# Linear Regression
# By Tianran Zhang, 2017-10-08.

basicdata <- read.table(file = 'basicData.txt', header = T)
x <- basicdata$X
y <- basicdata$y
d <- dim(basicdata)

model.predict <- function(x, y) {
  ymean <- mean(y)
  xmean <- mean(x)
  
  d1 <- 0
  d2 <- 0
  for (i in d[1]) {
    d1 <- d1 + (x[i] - xmean) * (y[i] - ymean)
    d2 <- d2 + (x[i] - xmean) ^ 2
  }
  
  c1 <<- d1 / d2
  c0 <<- ymean - c1 * xmean
  
  x * c1 + c0
}

yhat <- model.predict(x, y)
TrainError <- sum((y - yhat) ^ 2) / d[1]

xtest <- basicdata$Xtest
ytest <- basicdata$Ytest

ytesthat <- xtest * c1 + c0
TestError <- sum((ytest - ytesthat) ^ 2) / d[1]

print(TrainError)
print(TestError)

plot(
  x,
  yhat,
  type = 'l',
  xlim = c(-10, 10),
  ylim = c(-300, 400),
  col = 'green',
  main = "Training Data"
)
points(x, y, pch = 20, col = 'blue')
