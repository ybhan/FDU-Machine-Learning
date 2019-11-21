# Ridge Regression
# By Tianran Zhang, 2017-10-08.
# Modified by Yuanbo Han, 2019-11-20.

d <- read.table(file = 'prostate.data.txt', head = TRUE)
nrow(d)
ncol(d)

d1 <- d
# Shuffle data
#set.seed(123)
#d1 <- d[sample(nrow(d)), ]

# Divide data into training (first 50 instances) and test (the rest) sets
x <- as.matrix(d1[1:50, 1:8])
y <- as.matrix(d1[1:50, 9])
xtest <- as.matrix(d1[51:nrow(d1), 1:8])
ytest <- as.matrix(d1[51:nrow(d1), 9])

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

# Function to compute theta of ridge regression
ridge <- function(x, y, delta2) {
  theta <- ginv(t(x) %*% x + diag(delta2, 8), tol = 0) %*% t(x) %*% y
  theta
}

theta <- matrix(nrow = 1000, ncol = 8)
TrainError <- matrix(ncol = 1000)
TestError <- matrix(ncol = 1000)

# Perform ridge regression with 1000 different delta^2
delta2 <- 10 ^ ((1:1000) * 0.006 - 2)

for (i in 1:1000) {
  theta[i, ] <- ridge(x, y, delta2[i])
  yhat <- x %*% theta[i, ]
  TrainError[i] <- sqrt(sum((y - yhat) ^ 2) / sum((y + ymean) ^ 2))
  ytesthat <- xtest %*% theta[i, ]
  TestError[i] <- sqrt(sum((ytest - ytesthat) ^ 2) / sum((ytest + ymean) ^ 2))
}

# Plot theta against delta^2
plot(
  delta2,
  theta[, 1],
  xlim = c(0.01, 10000),
  ylim = c(-0.2, 0.6),
  log = 'x',
  type = 'l',
  col = 1,
  ylab = expression(theta),
  xlab = expression(delta ^ 2),
  lwd = 2
)
for (i in 2:8) {
  lines(delta2, theta[, i], col = i, lwd = 2)
}
legend.text <- colnames(d1)[-9]
legend(
  'topright',
  legend = legend.text,
  col = c(1:8),
  lty = 1,
  cex = 0.6
)
grid(lty = 1)

# Plot errors against delta^2
plot(
  delta2,
  TrainError,
  ylim = c(0.2, 0.55),
  log = 'x',
  type = 'l',
  col = 1,
  lwd = 2,
  xlab = expression(delta ^ 2),
  ylab = "error"
)
lines(delta2, TestError, lwd = 2, col = 2)
legend(
  'bottomright',
  legend = c("Train", "Test"),
  col = c(1, 2),
  lty = 1,
  cex = 1
)
grid(lty = 1)
