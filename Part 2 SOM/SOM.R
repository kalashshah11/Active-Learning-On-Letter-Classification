library('kohonen')
library(mlbench)

# Function for color palette
coolBlueHotRed <- function(n, alpha = 1) {rainbow(n, end=4/6, alpha=alpha)[n:1]}

# Get Training Data
data("LetterRecognition")
lettr <- as.numeric(LetterRecognition$lettr)
data_input <- cbind(lettr, as.data.frame(scale(LetterRecognition[-1])))
data_train <- as.matrix(data_input)


# Generate Grid
grid <- somgrid(xdim=20, ydim=20, topo="hexagonal")

# SOM
som_model <- som(data_train, grid=grid, alpha=c(0.05,0.01), radius=0.001)

# Specify which column to plot
col <- 1
plot(som_model, type="property", property=data_train[,col], main=names(data_train)[col], palette.name=coolBlueHotRed)