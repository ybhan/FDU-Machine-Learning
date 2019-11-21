# Ridge Regression with K-Fold Cross Validation
# By Tianran Zhang, 2017-10-08.
# Modified by Yuanbo Han, 2019-11-21.

d <- read.table(file = 'prostate.data.txt', head = TRUE)

# Shuffle data
set.seed(123)
d1 <- d[sample(nrow(d)), ]

k <- 5  # for k-fold cross validation

# Create k folds with (virtually) equal size
folds <- cut(seq(1, nrow(d1)), breaks = k, labels = FALSE)

# Function to compute theta of ridge regression
ridge <- function(x, y, delta2) {
  theta <- ginv(t(x) %*% x + diag(delta2, 8), tol = 0) %*% t(x) %*% y
  theta
}

theta <- matrix(nrow = 1000, ncol = 8)
TrainError <- matrix(data = 0, ncol = 1000)
TestError <- matrix(data = 0, ncol = 1000)

delta2 <- 10 ^ ((1:1000) * 0.006 - 2)

for (num in 1:k) {
  fold_index <- which(folds == num, arr.ind = TRUE)
  
  # Divide data into training and test sets
  x <- as.matrix(d1[-fold_index, 1:8])
  y <- as.matrix(d1[-fold_index, 9])
  xtest <- as.matrix(d1[fold_index, 1:8])
  ytest <- as.matrix(d1[fold_index, 9])
  
  # Mean and variance of training data
  xmean <- colMeans(x)
  xsd <- apply(x, 2, sd)
  ymean <- colMeans(y)
  
  standardize <- function(z) {
    (z - xmean) / xsd
  }
  
  # Standardize all data by training data
  x <- t(apply(x, 1, standardize))
  xtest <- t(apply(xtest, 1, standardize))
  y <- y - ymean
  ytest <- ytest - ymean
  
  # Perform ridge regression with 1000 different delta^2
  for (i in 1:1000) {
    theta[i,] <- ridge(x, y, delta2[i])
    yhat <- x %*% theta[i,]
    TrainError[i] <-
      TrainError[i] + sqrt(sum((y - yhat) ^ 2) / sum((y + ymean) ^ 2))
    ytesthat <- xtest %*% theta[i, ]
    TestError[i] <-
      TestError[i] + sqrt(sum((ytest - ytesthat) ^ 2)
                          / sum((ytest + ymean) ^ 2))
  }
}

TrainError <- TrainError / k
TestError <- TestError / k

# Plot errors against delta^2
plot(
  delta2,
  TrainError,
  ylim = c(0.2, 0.45),
  log = 'x',
  type = 'l',
  col = 1,
  lwd = 2,
  xlab = expression(delta ^ 2),
  ylab = 'error'
)
lines(delta2, TestError, col = 2, lwd = 2)
legend(
  'bottomright',
  legend = c('Train', 'Test'),
  col = c(1, 2),
  lty = 1,
  cex = 1
)
grid()
