# Exercise of Data Structure in R
# Edited by Yuanbo Han, Oct. 20, 2017.

# Exercise 1
v1 <- character()
v2 <- double()
v3 <- logical()
v4 <- integer()
v5 <- complex()
v6 <- raw()

# Exercise 2
is.vector(v1, mode = "character")
is.vector(v2, mode = "double")
is.vector(v3, mode = "logical")
is.vector(v4, mode = "integer")
is.vector(v5, mode = "complex")
is.vector(v6, mode = "raw")

# Exercise 3
l <- list(v2,v1,v3)
print(l)

# Exercise 4
m1 <- matrix(1:12, nrow=3, ncol=4, byrow=TRUE)

# Exercise 5
m2 <- matrix(runif(12), nrow=3, ncol=4, byrow=FALSE)

# Exercise 6
m3 <- diag(3, nrow=2, ncol=2)
m4 <- matrix(seq(1,7,2), nrow=2, ncol=2, byrow=TRUE)
m3 + m4
m3 - m4
m3 * m4
m3 / m4
m3 %*% m4
t(m4) # == aperm(m4, c(2,1)) == aperm(m4)
diag(m4)
solve(m4) # The inverse of m4.
eigen(m4)
svd(m3)
qr(m4)
m3 %o% m4 # == outer(m3,m4,"*")

# Exercise 7
m5 <- matrix(sample(1:1000,size=15,replace=TRUE),nrow=5)
print(m5)
print(t(m5))

# Exercise 8
Jeff <- data.frame(name="Jeff", age=20)
Ran <- data.frame(name="Ran", age=18)
A <- data.frame(name="A", age=15)
B<- data.frame(name="B", age=16)
C <- data.frame(name="C", age=17)

d <- rbind(Jeff, Ran, A, B, C)
d

# Exercise 9
summary(d)

# Exercise 10
d[1,]
d[length(d),]
d$name
