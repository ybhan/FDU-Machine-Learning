# Polynomial Regression
# By Tianran Zhang, 2017-10-08.
# Modified by Yuanbo Han, 2019-11-20.

#install.packages("MASS")
library(MASS)

basicdata <- read.table(file = 'basicData.txt', header = T)
x <- basicdata$X
y <- basicdata$y
xtest <- basicdata$Xtest
ytest <- basicdata$Ytest

d <- dim(basicdata)

makeXpoly <- function(x, deg) {
  m <- matrix(data = NA,
              ncol = deg + 1,
              nrow = d)
  for (i in (0:deg)) {
    m[, i + 1] <- x ^ i
  }
  m
}

leastSquaresBasis <- function(x, y, deg) {
  Xpoly <- makeXpoly(x, deg)
  b <<- ginv(t(Xpoly) %*% Xpoly, tol = 0) %*% t(Xpoly) %*% y
  
  Xpoly %*% b
}

for (i in 0:10) {
  yhat <- leastSquaresBasis(x, y, i)
  plot(
    x,
    yhat,
    type = 'l',
    xlim = c(-10, 10),
    ylim = c(-300, 400),
    col = 'red',
    main = c("Training data", paste("deg=", i, sep = '')),
    lwd = 5
  )
  points(
    x,
    y,
    pch = 20,
    col = 'blue',
    cex = 0.3
  )
  
  TrainError <- sum((y - yhat) ^ 2) / d[1]
  ytesthat <- makeXpoly(xtest, i) %*% b
  TestError <- sum((ytest - ytesthat) ^ 2) / d[1]
  
  print(paste("k =", i))
  print(paste("TrainError:", TrainError, "TestError:", TestError))
}
