install.packages("ggplot2")

# Load ggplot2 library
library(ggplot2)

# Load diamonds dataset
data(diamonds)

# View the dataset
View(diamonds)

# Example usage of qplot() to create a scatterplot
qplot(carat, price, data = diamonds)

# Example of transforming variables using log function
qplot(log(carat), log(price), data = diamonds)

# Create a smaller dataset from diamonds
set.seed(1000) # Make the sample reproducible
dsmall <- diamonds[sample(nrow(diamonds), 100),]

# Investigate relationship between variables using color aesthetic
qplot(log(carat), log(price), data = dsmall, colour = color)

# Investigate relationship between variables using shape aesthetic
qplot(log(carat), log(price), data = dsmall, shape = cut)

# Exercise 1
# Apply size aesthetic to investigate relationship between variables in dsmall dataset
qplot(log(carat), log(price), data = dsmall, size = clarity)

# Exercise 2
# Use bar geoms to show color distribution of diamonds in dsmall dataset
qplot(color, data = dsmall, geom = "bar")

# Change plot type using geom
qplot(log(carat), log(price), data = dsmall, geom = "line")

# Use jittered geom
qplot(color, price / carat, data = diamonds, geom = "jitter")

# Use boxplot geom
qplot(color, price / carat, data = diamonds, geom = "boxplot")

# Use histogram geom
qplot(carat, data = diamonds, geom = "histogram")

# Use density geom
qplot(carat, data = diamonds, geom = "density")

# Compare distributions with color aesthetic
qplot(carat, data = diamonds, geom = "density", colour = color)

# Fill different distributions with colors and use histogram plot
qplot(carat, data = diamonds, geom = "density", fill = color)

# Use qplot options to control appearance
qplot(carat, price, data = dsmall, xlab = "Price ($)", ylab = "Weight (carats)", main = "Price-weight relationship")


#smoothing the plot:
qplot(log(carat), log(price), data = dsmall, geom = "smooth")

#combine multiple geoms to smooth
qplot(log(carat), log(price), data = dsmall, geom = c("point", "smooth"))
