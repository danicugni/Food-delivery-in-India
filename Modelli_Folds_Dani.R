rm(list=ls())
setwd("C:/Users/gaiap/OneDrive/Desktop/Università/2022-2023/Aziendali/Progetto")
source("funzioni_classificazione.R")
load("dati_rinominati.RData")
#load("modelli_continui.RData")

#load("modelli_continui2.RData")

# Creazione dei folds e quantità necessarie
load("folds.RData")
load("foldid.RData")

params <- list()
accuracy.matrix <- matrix(NA, nrow=1, ncol=2)
colnames(accuracy.matrix) <- c("Accuracy", "Standard Error")

# Modifica e standardizzazione

dati$Latitude <- scale(dati$Latitude)
dati$Longitude <- scale(dati$Longitude)
dati$Age <- scale(dati$Age)
dati$Monthly_Income <- factor(dati$Monthly_Income, ordered=F)
names(dati)[13] = "Ease_and_Convenience"
names(dati)[17] = "More_Offers_and_Discounts"

# Funzioni varie 
# distanza minima da (0,1)
library(ModelMetrics)
cz = function(modello, soglia, y = ver$y) {
  x = 1- ModelMetrics::specificity(y,modello,cutoff = soglia)
  y = ModelMetrics::sensitivity(y,modello,cutoff = soglia)
  # distanza da (0,1)
  sqrt(((0-x)^2 + (1-y)^2))
}


# Distanza massima dalla bisettrice
cb = function(modello, soglia, y = ver$y) {
  x = 1- specificity(y,modello,cutoff = soglia)
  y = sensitivity(y,modello,cutoff = soglia)
  abs(x-y) * sqrt(2)/2
}


probs = seq(0.1,0.9, 0.05)
ss




# Creazione delle matrici del disegno

ids.leak = which(names(dati)=="y")
X = model.matrix(~., data = dati[, -ids.leak])
X_int = model.matrix(y ~ .^2, data=dati)

y.num = rep(0,length(dati$y))
y.num[which(dati$y=='Yes')]=1
y.num[which(dati$y=='No')]=0


# Variabili da togliere
var_null <- c("Time_Saving", "Self_Cooking", "Unaffordable", "Educational_Qualification",
              "Good_Tracking_System", "Late_Delivery")

var_null_id <- which(colnames(dati)  %in%   var_null)




# LOGISTICO ---------

# 1. Modello logistico--------------------------------------------------------------
mlog = glm(dati$y~., data = dati[,-var_null_id], family = binomial)
summary(mlog)

# 2. Modello logistico stepwise --------
m0 = glm(y~1, data = dati[,-var_null_id], family = binomial, control = list( maxit=100 ))
mlog21 = step(m0, scope=formula(mlog), direction = "forward")
summary(mlog21)
var_logistico <- names(coef(mlog21))[-1]
pred <- mlog21$fitted.values
hist(pred, nclass = 20)



#per il confronto con gli altri modelli
K = 4
err_log <- rep(NA, K)
thres_log <- rep(NA, K)
#par(mfrow = c(2,2))


set.seed(1)
for(i in 1:K) {
  mlog1 = glm(y ~ 1, weights = NULL, data = dati[foldid != i, -var_null_id], 
              control = list(maxit = 100), family = "binomial")
  
  mlog2 = step(mlog1, scope = formula(mlog), direction = "forward", trace = 0) #direction = "both" 
  mlog2.pred = predict(mlog2, newdata = dati[foldid == i, -var_null_id], type = "response")
  print(summary(mlog2))
  hist(mlog2.pred)
  thres = probs[which.min(sapply(probs, function(l) cz(mlog2.pred, l, y=y.num[foldid ==i])))]
  thres_log[i] <- thres
  pp <- as.numeric(as.numeric(mlog2.pred) > thres)
  t <- tabella.sommario(pp, dati$y[foldid ==i])
  err_log[i] <- 1- sum(diag(t))/sum(t)
}

err_log
logistic.e <- mean(err_log)
logistic.sd <- sd(err_log)


accuracy.matrix[1,1] <- 1-logistic.e
accuracy.matrix[1,2] <- logistic.sd

rownames(accuracy.matrix) = c("Logistico Stepwise CV")

params$Logistico <- mean(thres_log)





# LASSO ---------------
library(caret)
grid = 10^seq(-5, 5, length=500)
lasso_caret_acc <- train(y ~ ., data=dati, metric="Accuracy", method="glmnet",
                         trControl = trainControl(method = "repeatedcv", number=4,# repeats = 10, 
                                                  verboseIter = T, 
                                                  classProbs = T,#search = "random", 
                                                  index = folds, savePredictions = "all"),
                         tuneGrid=expand.grid(alpha = 1, lambda = grid))

####
probs <- seq(.1, 0.9, by = 0.05)
ths_lasso <- thresholder(lasso_caret_acc,
                         threshold = probs,
                         final = F,
                         statistics = "Accuracy")
accuracy.matrix <- rbind(accuracy.matrix,max(ths_lasso$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Lasso")
params$Lasso <- ths_lasso[which.max(ths_lasso$Accuracy),]


####
library(glmnet)
grid 
probs 
err.lasso = rep(NA, K)
thres_lasso = rep(NA, K)

set.seed(1)
for(i in 1:K){
  cat("Fold ",i,"/",K,"\n")
  lasso = glmnet(X[foldid != i,-1], dati$y[foldid != i], alpha = 1, lambda = params$Lasso$lambda, 
                 standardize = FALSE, family = "binomial")
  #Selezione manuale di lambda
  lasso.pred = predict(lasso, newx = X[foldid == i,-1], type = "response")
  thres = probs[which.min(sapply(probs, function(l) cz(lasso.pred, l, y=y.num[foldid ==i])))]
  thres_lasso[i] <- thres
  pp <- as.numeric(as.numeric(lasso.pred) > thres)
  t <- tabella.sommario(pp, dati$y[foldid ==i])
  err.lasso[i] <- 1- sum(diag(t))/sum(t)
}


err.lasso
cat("CV Error: ", mean(err.lasso), "(sd: ", sd(err.lasso),")\n")
accuracy.matrix[nrow(accuracy.matrix),2] <- sd(err.lasso)



lasso = glmnet(X[,-1], dati$y, alpha = 1, lambda = grid, standardize = FALSE, family="binomial")
plot(lasso, xvar = "lambda", label = T)

lasso = glmnet(X[,-1], dati$y, alpha = 1, lambda = params$Lasso$lambda, standardize = FALSE,family="binomial")
plot(lasso, xvar = "lambda", label = T)

coef(lasso, s = params$Lasso$lambda)

varlasso <- colnames(X)[which(coef(lasso, s = params$Lasso$lambda) != 0)]
varlasso
length(varlasso)





# LASSO CON INTERAZIONI ---------------
grid = 10^seq(-5, 5, length=500)
lasso_int_caret_acc <- train(x = X_int, y = dati$y, metric="Accuracy", method="glmnet",
                             trControl = trainControl(method = "repeatedcv", number=4,# repeats = 10, 
                                                      verboseIter = T, 
                                                      classProbs = T,#search = "random", 
                                                      index = folds, savePredictions = "all"),
                             tuneGrid=expand.grid(alpha = 1, lambda = grid))

####
probs <- seq(.1, 0.9, by = 0.05)
ths_lasso_int <- thresholder(lasso_int_caret_acc,
                             threshold = probs,
                             final = F,
                             statistics = "Accuracy")
accuracy.matrix <- rbind(accuracy.matrix,max(ths_lasso_int$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Lasso con interazioni")
params$Lasso_int <- ths_lasso_int[which.max(ths_lasso_int$Accuracy),]


####
library(glmnet)
grid 
probs 
err.lasso.int = rep(NA, K)
thres_lasso_int = rep(NA, K)

set.seed(1)
for(i in 1:K){
  cat("Fold ",i,"/",K,"\n")
  lasso = glmnet(X_int[foldid != i,-1], dati$y[foldid != i], alpha = 1, lambda = params$Lasso_int$lambda, 
                 standardize = FALSE, family = "binomial")
  #Selezione manuale di lambda
  lasso.pred = predict(lasso, newx = X_int[foldid == i,-1], type = "response")
  thres = probs[which.min(sapply(probs, function(l) cz(lasso.pred, l, y=y.num[foldid ==i])))]
  thres_lasso_int[i] <- thres
  pp <- as.numeric(as.numeric(lasso.pred) > thres)
  t <- tabella.sommario(pp, dati$y[foldid ==i])
  err.lasso.int[i] <- 1- sum(diag(t))/sum(t)
}


err.lasso.int
cat("CV Error: ", mean(err.lasso.int), "(sd: ", sd(err.lasso.int),")\n")
accuracy.matrix[nrow(accuracy.matrix),2] <- sd(err.lasso.int)



lasso_int = glmnet(X_int[,-1], dati$y, alpha = 1, lambda = grid, standardize = FALSE, family="binomial")
plot(lasso_int, xvar = "lambda", label = T)

lasso_int = glmnet(X_int[,-1], dati$y, alpha = 1, lambda = params$Lasso$lambda, standardize = FALSE,family="binomial")
plot(lasso_int, xvar = "lambda", label = T)

coef(lasso_int, s = params$Lasso_int$lambda)

varlasso_int <- colnames(X_int)[which(coef(lasso_int, s = params$Lasso_int$lambda) != 0)]
varlasso_int
length(varlasso_int)




# RIDGE ---------------
grid = 10^seq(-5, 5, length=500)
ridge_caret_acc <- train(y ~ ., data=dati, metric="Accuracy", method="glmnet",
                         trControl = trainControl(method = "repeatedcv", number=4,# repeats = 10, 
                                                  verboseIter = T, 
                                                  classProbs = T, #search = "random", 
                                                  index = folds, savePredictions = "all"#,
                                                  #summaryFunction = twoClassSummary
                         ),
                         tuneGrid=expand.grid(alpha = 0, lambda = grid))

probs <- seq(.1, 0.9, by = 0.05)
ths_ridge <- thresholder(ridge_caret_acc,
                         threshold = probs,
                         final = F,
                         statistics = "Accuracy")
accuracy.matrix <- rbind(accuracy.matrix, max(ths_ridge$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Ridge")
params$Ridge <- ths_ridge[which.max(ths_ridge$Accuracy),]



#########

grid 
soglia 
err.ridge = rep(NA,K)
thres_ridge = rep(NA, K)

set.seed(1)
for(i in 1:K){
  cat("Fold ",i,"/",K,"\n")
  ridge = glmnet(X[foldid != i,-1], dati$y[foldid != i], alpha = 0, lambda = params$Ridge$lambda, 
                 standardize = FALSE, family = "binomial", type.measure="class")
  #Selezione manuale di lambda
  ridge.pred = predict(ridge, newx = X[foldid == i,-1], type = "response")
  thres = probs[which.min(sapply(probs, function(l) cz(ridge.pred, l, y=y.num[foldid ==i])))]
  pp <- as.numeric(as.numeric(ridge.pred) > thres)
  t <- tabella.sommario(pp, dati$y[foldid ==i])
  err.ridge[i] <- 1- sum(diag(t))/sum(t)
}

err.ridge
cat("CV Error: ", mean(err.ridge), "(sd: ", sd(err.ridge),")\n")
accuracy.matrix[nrow(accuracy.matrix),2] <- sd(err.ridge)




ridge = glmnet(X[,-1], dati$y, alpha = 0, lambda = grid, standardize = FALSE, family="binomial")
plot(ridge, xvar = "lambda", label = T)

ridge = glmnet(X[,-1], dati$y, alpha = 0, lambda = params$Ridge$lambda, standardize = FALSE,family="binomial")
plot(ridge, xvar = "lambda", label = T)

coef(ridge, s = params$Ridge$lambda)





# RIDGE INTERAZIONI ---------------
grid = 10^seq(-5, 5, length=500)
ridge_caret_acc_int <- train(X_int[,-1], dati$y, metric="Accuracy", method="glmnet",
                         trControl = trainControl(method = "repeatedcv", number=4,# repeats = 10, 
                                                  verboseIter = T, 
                                                  classProbs = T, #search = "random", 
                                                  index = folds, savePredictions = "all"#,
                                                  #summaryFunction = twoClassSummary
                         ),
                         tuneGrid=expand.grid(alpha = 0, lambda = grid))

probs <- seq(.1, 0.9, by = 0.05)
ths_ridge_int <- thresholder(ridge_caret_acc_int,
                         threshold = probs,
                         final = F,
                         statistics = "Accuracy")
accuracy.matrix <- rbind(accuracy.matrix, max(ths_ridge_int$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Ridge con interazioni")
params$Ridge_int <- ths_ridge_int[which.max(ths_ridge_int$Accuracy),]



#########

grid 

err.ridge.int = rep(NA,K)
thres_ridge.int = rep(NA, K)

set.seed(1)
for(i in 1:K){
  cat("Fold ",i,"/",K,"\n")
  ridge = glmnet(X_int[foldid != i,-1], dati$y[foldid != i], alpha = 0, lambda = params$Ridge$lambda, 
                 standardize = FALSE, family = "binomial", type.measure="class")
  #Selezione manuale di lambda
  ridge.pred = predict(ridge, newx = X_int[foldid == i,-1], type = "response")
  thres = probs[which.min(sapply(probs, function(l) cz(ridge.pred, l, y=y.num[foldid ==i])))]
  pp <- as.numeric(as.numeric(ridge.pred) > thres)
  t <- tabella.sommario(pp, dati$y[foldid ==i])
  err.ridge.int[i] <- 1- sum(diag(t))/sum(t)
}

err.ridge
cat("CV Error: ", mean(err.ridge), "(sd: ", sd(err.ridge),")\n")
accuracy.matrix[nrow(accuracy.matrix),2] <- sd(err.ridge)




ridge_int = glmnet(X_int[,-1], dati$y, alpha = 0, lambda = grid, standardize = FALSE, family="binomial")
plot(ridge, xvar = "lambda", label = T)

ridge_int = glmnet(X_int[,-1], dati$y, alpha = 0, lambda = params$Ridge$lambda, standardize = FALSE,family="binomial")
plot(ridge, xvar = "lambda", label = T)

coef(ridge_int, s = params$Ridge$lambda)






# GBM ------

#######
# gbm_caret_roc <- train(y ~ ., data=dati, metric="Accuracy", method="gbm",
#                        trControl = trainControl(method="repeatedcv", number=4,# repeats = 10,
#                                                 verboseIter=T,
#                                                 classProbs = T,#search = "random",
#                                                 index = multiFoldIndices, savePredictions = "all"#,
#                                                 #summaryFunction = twoClassSummary
#                                                 ),
#                        tuneGrid=expand.grid(n.trees=c(500, 1000, 1500, 3000), interaction.depth=c(1,2,4,6),
#                                            shrinkage=c(0.001,0.1), n.minobsinnode=1))

########
library(gbm)
gbm_caret_acc <- train(y ~ ., data=dati, metric="Accuracy", method="gbm",
                       trControl = trainControl(method = "repeatedcv", number=4,# repeats = 10,
                                                verboseIter = T,
                                                classProbs = T,#search = "random",
                                                index = folds, savePredictions = "all"),
                       tuneGrid=expand.grid(n.trees=c(500, 1000, 1500, 3000), interaction.depth=c(2,4,6),
                                            shrinkage=c(0.001,0.1), n.minobsinnode=1))
probs <- seq(.1, 0.9, by = 0.05)
ths_gbm <- thresholder(gbm_caret_acc,
                       threshold = probs,
                       final = F,
                       statistics = "Accuracy")
accuracy.matrix <- rbind(accuracy.matrix, max(ths_gbm$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Gradient Boosting")
params$Gradient_Boosting <- ths_gbm[which.max(ths_gbm$Accuracy),]


err.gbm <- rep(NA, K)
set.seed(123)

for(i in 1:K) {
  # boost.delivery <- gbm((as.numeric(y)-1) ~ ., data = dati[foldid != i,],
  #                       distribution="bernoulli", n.trees= params$Gradient_Boosting$n.trees, 
  #                       interaction.depth=params$Gradient_Boosting$interaction.depth, 
  #                       shrinkage=params$Gradient_Boosting$shrinkage)
  boost.delivery <- gbm(y.num[foldid != i] ~ ., data=dati[foldid != i,-ids.leak], 
                        distribution="bernoulli", 
                        n.trees=params$Gradient_Boosting$n.tree, 
                        interaction.depth=params$Gradient_Boosting$interaction.depth, 
                        shrinkage=params$Gradient_Boosting$shrinkage, 
                        n.cores = 10, keep.data = F)
  yhat.boost <- predict(boost.delivery, newdata = dati[foldid == i,] , 
                        type = "response", n.trees=params$Gradient_Boosting$n.tree)
  err <- tabella.sommario(yhat.boost > params$Gradient_Boosting$prob_threshold, dati$y[foldid ==i])
  err.gbm[i] = 1 - sum(diag(err))/sum(err)
  
}

err.gbm

cat("CV Error: ", mean(err.gbm), "(sd: ", sd(err.gbm),")\n")
accuracy.matrix[nrow(accuracy.matrix),2] <- sd(err.gbm)



# interpretazione
library(gbm)
gb <- gbm(y.num ~ ., data=dati[,-ids.leak], distribution="bernoulli", 
          n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
          interaction.depth=params$Gradient_Boosting$interaction.depth, 
          shrinkage=params$Gradient_Boosting$shrinkage, 
          n.cores = 10, keep.data = F)


# Grafici delle variabili più importanti

summary(gb) 
mai.old<-par()$mai
mai.old
mai.new<-mai.old
mai.new[2] <- 2.1 
mai.new
par(mai=mai.new)
summary(gb, las=1) 
summary(gb, las=1, cBar=10) 
par(mai=mai.old)

plot(gb, n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
     interaction.depth=params$Gradient_Boosting$interaction.depth, 
     shrinkage=params$Gradient_Boosting$shrinkag)
plot(gb, i.var=13, n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
     interaction.depth=params$Gradient_Boosting$interaction.depth, 
     shrinkage=params$Gradient_Boosting$shrinkag)
plot(gb, i.var=5, n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
     interaction.depth=params$Gradient_Boosting$interaction.depth, 
     shrinkage=params$Gradient_Boosting$shrinkag)
plot(gb, i.var=c(1,5), n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
     interaction.depth=params$Gradient_Boosting$interaction.depth, 
     shrinkage=params$Gradient_Boosting$shrinkag) #bivariata
#
plot(gb, i.var=3,n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
     interaction.depth=params$Gradient_Boosting$interaction.depth, 
     shrinkage=params$Gradient_Boosting$shrinkag) # variabile indicatrice
plot(gb, i.var=6, n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
     interaction.depth=params$Gradient_Boosting$interaction.depth, 
     shrinkage=params$Gradient_Boosting$shrinkag)

plot(gb, i=23, n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
     interaction.depth=params$Gradient_Boosting$interaction.depth, 
     shrinkage=params$Gradient_Boosting$shrinkag)# variabile qualitativa
plot(gb, i=17, n.trees=params$Gradient_Boosting$n.trees, cv.folds = 4, 
     interaction.depth=params$Gradient_Boosting$interaction.depth, 
     shrinkage=params$Gradient_Boosting$shrinkag) #variabile che non ha effetto




# RANDOM FOREST -----------------------
tr <- trainControl(method="repeatedcv", number=4,
                   verboseIter=T, 
                   classProbs = T, #search = "random", 
                   index = folds, savePredictions = "all"#, 
                   #summaryFunction = twoClassSummary
)
tuneGr <- expand.grid(mtry=c(1, 2, seq(3, (ncol(dati)-1), 1)))

rf_caret_acc <- train(y ~ ., data=dati, metric="Accuracy", method="rf",
                      trControl = tr,
                      tuneGrid=tuneGr)
probs <- seq(.1, 0.9, by = 0.05)
ths_rf <- thresholder(rf_caret_acc,
                      threshold = probs,
                      final = F,
                      statistics = "Accuracy")
accuracy.matrix <- rbind(accuracy.matrix, max(ths_rf$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Random Forest")
params$Random_Forest <- ths_rf[which.max(ths_rf$Accuracy),]


###########
library(randomForest)
nn = 500
colnum
soglia
err.rf = rep(NA, K)
set.seed(1)
for(i in 1:K){
  cat("Fold ",i,"/",K,"\n")
  rf = randomForest(y ~ ., data = dati[foldid != i,],
                    mtry = params$Random_Forest$mtry, ntree = nn)
  pred= predict(rf, newdata = dati[foldid ==i, ], type = "prob")[,2]
  rf.tab <- tabella.sommario(pred > params$Random_Forest$prob_threshold , dati$y[foldid == i])
  err.rf[i] = 1- sum(diag(rf.tab))/sum(rf.tab)
}

mean(err.rf)
cat("CV Error: ", 1-mean(err.rf), "(sd: ", sd(err.rf),")\n")
accuracy.matrix[nrow(accuracy.matrix),2] <- sd(err.rf)



########
library(randomForest)
set.seed(2222)
rf = randomForest(x = dati[, -ids.leak], y = as.factor(dati$y), ntree = 500,
                  mtry = params$Random_Forest$mtry, importance = TRUE)
plot(rf, main = "OOB")
legend(250, 0.76 ,c("errore", "classe 0", "classe 1" ), col=1:3, lty=1:3)
rf
varImpPlot(rf)





# ALBERO --------
library(rpart)

fitControl <- trainControl(method = "repeatedcv",
                           number = 4,
                           verboseIter=T, 
                           classProbs = T, #search = "random", 
                           index = folds, savePredictions = "all")
tuneGr <- expand.grid(maxdepth=3:30)
tree_caret_acc <- train(y ~ ., data=dati, 
                        method = "rpart2", metric="Accuracy", trControl = fitControl, 
                        tuneGrid=tuneGr)

probs <- seq(.2, 0.9, by = 0.05)
ths_tree <- thresholder(tree_caret_acc,
                        threshold = probs,
                        final = F,
                        statistics = "Accuracy")
accuracy.matrix <- rbind(accuracy.matrix, max(ths_tree$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Albero")
params$Albero <- ths_tree[which.max(ths_tree$Accuracy),]


# cv

complexity
soglia
set.seed(1)
err.tree = rep(NA, K)
for(i in 1:K){
  cat("Fold ",i,"/",K,"\n")
  classtree <- rpart(y ~ ., data=dati[foldid!=i,], method = "class",
                     control = rpart.control(maxdepth= params$Albero$maxdepth
                     ))
  pp = predict(classtree, newdata = dati[foldid == i,])[,2]
  pp <- as.numeric(pp >  params$Albero$prob_threshold
  )
  t <- tabella.sommario(pp, dati$y[foldid ==i])
  err.tree[i] <- 1- sum(diag(t))/sum(t)
}
err.tree
cat("CV Error: ", mean(err.tree), "(sd: ", sd(err.tree),")\n")
accuracy.matrix[nrow(accuracy.matrix),2] <- sd(err.tree)









classtree <- rpart(y ~ ., data=dati, method = "class",
                   control = rpart.control(maxdepth=params$Albero$maxdepth))

x11()
plot(classtree, uniform=T)
text(classtree, pretty = 4, cex = 1)
library(rattle)
library(rpart.plot)
fancyRpartPlot(classtree,yesno=2,split.col="black",nn.col="black", 
               caption="",palette="Set3",branch.col="black")

fancyRpartPlot(classtree)
rpart.plot(classtree, box.palette="BuGn", #shadow.col="gray", 
           nn=TRUE, type=5)




# BOOSTING ----------
library(ada)
niter = round(seq(100, 2000, length.out=4))
err.boost = matrix(NA, nrow = length(niter), ncol = K)
ths_boost <- matrix()
n.cores <- parallel::detectCores()
registerDoParallel(cores = n.cores-10)

probs = seq(.1, .9, by=0.1)
for(i in 1:length(niter)){
  for(j in 1:K){
    test <- dati[foldid!=j,]
    train <- dati[foldid==j,]
    ada = ada(factor(train$y)~., data=train[,-ids.leak],
              #test.x = test[, -ids.leak], test.y = as.factor(test$y), 
              iter = niter[i], loss="logistic", 
              rpart.control(maxdepth=1, cp=-1, minsplit=0,xval=0))
    pred = predict(ada, newdata = test, type = "prob")[,2]
    thres = probs[which.min(sapply(probs, 
                                   function(l) cz(pred, l, y=y.num[foldid == j]
                                                  )))]
    boost.falsi = falsi(pred > thres, test$y)
    err.boost[i,j]= boost.falsi[1]
  }
  cat(i, "")
}

err.boost
best_iter <- which.min(apply(err.boost,1,mean))
best_iter 
cat("CV Error: ", mean(err.boost[best_iter,]), "( sd: ", sd(err.boost[best_iter,]),")\n")


accuracy.matrix <- rbind(accuracy.matrix, c(1-mean(err.boost[best_iter,], sd(err.boost[best_iter,]))))
accuracy.matrix[nrow(accuracy.matrix),2] <-sd(err.boost[best_iter,]) 
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Boosting")
params$Boosting <- niter[best_iter]




plot(niter, apply(err.boost, 1, mean), type = "b", xlab = "Numero di iterazioni",
     ylab = "Tasso di errata classificazione", main = "Boosting", ylim = c(0.01,0.17))
points(niter,apply(err.boost,1,mean) + apply(err.boost,1,sd), pch = "-")
points(niter, apply(err.boost,1,mean) - apply(err.boost,1,sd), pch = "-")
segments(x0 = niter, y0 = apply(err.boost,1,mean) - apply(err.boost,1,sd), 
         x1 = niter, y1 = apply(err.boost,1,mean) + apply(err.boost,1,sd))
#abline(h = min(apply(err.boost,1,mean)), col = 2, lty = 2)
points(niter[which.min(apply(err.boost,1,mean))], min(apply(err.boost,1, mean)), 
       col = 2, lwd = 3)

set.seed(2222)
ada = ada(factor(dati$y)~., data=dati[,-ids.leak],
          iter = niter[best_iter], 
          loss="logistic", 
          rpart.control(maxdepth=1, cp=-1, minsplit=0, xval=0))
summary(ada)
varplot(ada)







# BOOSTING CON INTERAZIONI ------------

library(ada)
niter = round(seq(100, 2000, length.out=4))
err.boost.int = matrix(NA, nrow = length(niter), ncol = K)
ths_boost.int <- matrix()
n.cores <- parallel::detectCores()
registerDoParallel(cores = n.cores-10)

for(i in 1:length(niter)){
  for(j in 1:K){
    test <- dati[foldid!=j,]
    train <- dati[foldid==j,]
    ada = ada(factor(train$y)~., data=train[,-ids.leak],
              #test.x = test[, -ids.leak], test.y = as.factor(test$y), 
              iter = niter[i], loss="logistic", 
              rpart.control(maxdepth=2, cp=-1, minsplit=0,xval=0))
    pred = predict(ada, newdata = test, type = "prob")[,2]
    thres = probs[which.min(sapply(probs, 
                                   function(l) cz(pred, l, y=y.num[foldid == j]
                                   )))]
    boost.falsi = falsi(pred > thres, test$y)
    err.boost.int[i,j]= boost.falsi[1]
    cat(i, "")
}
}
err.boost.int
best_iter <- which.min(apply(err.boost.int,1,mean))
best_iter 
cat("CV Error: ", mean(err.boost.int[best_iter,]), "( sd: ", sd(err.boost.int[best_iter,]),")\n")


accuracy.matrix <- rbind(accuracy.matrix, c(1-mean(err.boost.int[best_iter,], sd(err.boost.int[best_iter,]))))
accuracy.matrix[nrow(accuracy.matrix),2] <-sd(err.boost.int[best_iter,]) 
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Boosting con interazioni")
params$Boosting_Interazioni <- niter[best_iter]






# MARS -------------
fitControl <- trainControl(method = "repeatedcv",
                           number = 4,
                           verboseIter=T, 
                           classProbs = T,# search = "random", 
                           index = folds, savePredictions = "all", 
                           #summaryFunction = twoClassSummary
)
tuneGr <- expand.grid(#nprune = c(1,2,3),
  degree=c(1,2,3,4))

mars_caret_acc <- train(y ~ ., data=dati, 
                        method = "bagEarthGCV", metric="Accuracy", trControl = fitControl,
                        tuneGrid = tuneGr)

probs <- seq(.1, 0.9, by = 0.02)
# 
ths_mars <- thresholder(mars_caret_acc,
                        threshold = probs,
                        final = F,
                        statistics = "Accuracy")

probs <- seq(.1, 0.9, by = 0.05)
ths_mars <- thresholder(mars_caret_acc,
                        threshold = probs,
                        final = F,
                        statistics = "Accuracy")
# accuracy.matrix <- rbind(accuracy.matrix, max(ths_mars$Accuracy))
# rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("MARS")
params$MARS <- ths_mars[which.max(ths_mars$Accuracy),]


# in CV
library(earth)

err.mars <- rep(NA, K)
degree=c(1,2,3,4)
soglia

set.seed(1)

for(i in 1:K) {
  m_mars = earth(y ~ ., data = dati[foldid != i,], glm = list(family = binomial), 
                 thresh = 0.005, nk = 100, degree = params$MARS$degree)
  pred_m_mars = predict(m_mars, dati[foldid == i,], type = "response")
  pred_m_mars_class = ifelse(pred_m_mars > params$MARS$prob_threshold, "Yes", "No")
  mars.tab <- tabella.sommario(pred_m_mars_class, dati$y[foldid == i])
  err.mars[i] <- 1- sum(diag(mars.tab))/sum(mars.tab)
}

err.mars
mars.e <- mean(err.mars)
mars.sd <- sd(err.mars)
cat("CV Error: ", 1-mars.e," (sd:", mars.sd,")\n")

accuracy.matrix <- rbind(accuracy.matrix, c(1-mars.e, mars.sd))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("MARS")
params$MARS <- ths_mars[which.max(ths_mars$Accuracy), 1:2]



#Per l'interpretazione

m_mars = earth(y ~ ., data = dati, glm = list(family = binomial),
               thresh = 0.005, nk = 100, degree = params$MARS$degree)
summary(m_mars)
plotmo(m_mars)






# GAM --------------
library(gam)
#Per il confronto con gli altri modelli
scope = gam.scope(dati[foldid != i,-ids.leak], 
                  response = ids.leak, 
                  arg = c("df = 2", "df = 3", "df = 4"))#oppure stima.bal[ridotto, -ids.leak]
err.gam <- rep(NA, K)
set.seed(123)

for(i in 1:K) {
  gam1 = gam(y ~ 1, weights = NULL, family = binomial, data = dati[foldid != i,])
  scope = gam.scope(dati[foldid != i,-ids.leak], response = ids.leak, 
                    arg = c("df = 2", "df = 3", "df = 4"))#oppure stima.bal[ridotto, -ids.leak]
  gam.step = step.Gam(gam1, scope = scope)
  gam.step.pred = predict(gam.step, newdata = dati[foldid == i,], type = "response")
  indice <- which.min(sapply(probs, function(x) cz(modello = gam.step.pred, 
                                                   soglia = x, 
                                                   y = as.numeric(dati$y[foldid == i])-1)))  
  gam.step.tab = tabella.sommario(gam.step.pred > probs[indice], dati$y[foldid == i])
  err.gam[i] <- 1 - sum(diag(gam.step.tab))/sum(gam.step.tab)
}

err.gam
cat("CV Error: ", 1-mean(err.gam), "(sd: ", sd(err.gam),")\n")

accuracy.matrix <- rbind(accuracy.matrix, c(1-mean(err.gam), sd(err.gam)))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("GAM")



# LDA ------------

fitControl <- trainControl(method = "repeatedcv",
                           number = 4,
                           verboseIter=T, 
                           classProbs = T,# search = "random", 
                           index = folds, savePredictions = "all", 
                           #summaryFunction = twoClassSummary
)
lda_caret_acc <- train(y ~ ., data=dati, 
                       method = "lda", metric="Accuracy", trControl = fitControl)

probs <- seq(.1, 0.9, by = 0.02)


ths_lda <- thresholder(lda_caret_acc,
                       threshold = probs,
                       final = F,
                       statistics = "Accuracy")

accuracy.matrix <- rbind(accuracy.matrix, max(ths_lda$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("LDA")
params$LDA <- ths_lda[which.max(ths_lda$Accuracy),]



# LDA Sparsa ------------
fitControl <- trainControl(method = "repeatedcv",
                           number = 4,
                           verboseIter=T, 
                           classProbs = T,# search = "random", 
                           index = folds, savePredictions = "all"#, 
                           #summaryFunction = twoClassSummary
)
tuneGr <- expand.grid(NumVars=seq(0,ncol(dati)-1, by=1), lambda=seq(0,1, by=0.1))
slda_caret_acc <- train(y ~ ., data=dati, 
                        method = "sparseLDA", metric="Accuracy", trControl = fitControl,
                        tuneGrid=tuneGr)

probs <- seq(.1, 0.9, by = 0.02)
ths_slda <- thresholder(slda_caret_acc,
                        threshold = probs,
                        final = F,
                        statistics = "Accuracy")

accuracy.matrix <- rbind(accuracy.matrix, max(ths_slda$Accuracy))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Sparse LDA")
params$Sparse_LDA <- ths_slda[which.max(ths_slda$Accuracy),]



library(MASS)

soglia
err.slda <- rep(NA, K)
set.seed(123)

for(i in 1:K) {
  slda <- sda(X[foldid != i,-1], dati$y[foldid != i], lambda = params$Sparse_LDA$lambda, trace=T,
              stop = -params$Sparse_LDA$NumVars)
  
  slda.pred = predict(slda, newdata = X[foldid == i,-1])
  lda.tab = tabella.sommario(slda.pred$posterior[,2] > params$Sparse_LDA$prob_threshold, 
                             dati$y[foldid == i])
  err.slda[i] <- 1-sum(diag(lda.tab))/sum(lda.tab)
}

err.slda
slda.e <- mean(err.slda)
slda.sd <- sd(err.slda)
cat("CV Error: ", slda.e," (sd:", slda.sd,")\n")
accuracy.matrix[nrow(accuracy.matrix),2] <-  slda.sd
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Sparse LDA")


#
library(sparseLDA)
slda <- sda(X[,-1], dati$y, lambda = params$Sparse_LDA$lambda, trace=T,stop = -params$Sparse_LDA$NumVars)
summary(slda)
slda$varNames



sort(accuracy.matrix[,1])

save.image(file="modelli_dani.RData")






# Modelli ad effetti casuali -----------
dati2 <- read.csv("onlinedeliverydata.csv")
dati$Pin.code <- as.factor(dati2$Pin.code)
str(dati2$Pin.code)
#Pin code----
library(geosphere)
unici <- unique(dati2$Pin.code)
long.lat <- matrix(NA, nrow = 1, ncol =2)
colnames(long.lat) <- c("Longitudine", "Latitudine")
for(i in unici) {
  lat <- mean(dati$Latitude[dati2$Pin.code == i])
  long <- mean(dati$Longitude[dati2$Pin.code == i])
  long.lat <- rbind(long.lat, c(long,lat))
  rownames(long.lat)[nrow(long.lat)] <- i
}


long.lat <- long.lat[-1,]
long.lat
sum(table(dati2$Pin.code))
table(dati2$Pin.code)
unici[table(dati2$Pin.code) <2]
length(unici[table(dati2$Pin.code) <2])
code.piccoli <- names(which(table(dati2$Pin.code) < 2))
code.piccoli
length(code.piccoli)
code.grandi <- setdiff(rownames(long.lat), code.piccoli)
code.grandi
length(code.grandi)
dist.code.piccoli <- list()

for(i in 1:length(code.piccoli)){
  dist.code.piccoli[[i]] <- (distm(long.lat[which(rownames(long.lat) == code.piccoli[i]),], 
                                   long.lat[code.grandi,], fun = distHaversine))
  colnames(dist.code.piccoli[[i]]) <- code.grandi
}

vicini <- matrix(NA, nrow = 1, ncol = 1)
colnames(vicini) <- "A"


for(i in 1:length(code.piccoli)) {
  vicini <- rbind(vicini, code.grandi[which.min(dist.code.piccoli[[i]])])
  rownames(vicini)[nrow(vicini)] <- code.piccoli[i]
}
vicini <- vicini[-1,]
vicini

for(i in code.piccoli) {
  dati2$Pin.code[dati2$Pin.code == i] <- vicini[i]
}

Pin.code <- factor(dati2$Pin.code)

str(Pin.code)
#Modello ad intercetta casuale----

dati2 <- dati
dati2$Pin.code <- factor(Pin.code)
table(dati2$Pin.code)

dati$Pin.code = dati2$Pin.code


library(lme4)
var_logistico[6] <- "Educational_Qualification"
var_logistico[7] <- "Influence_of_Rating"
var_logistico <- var_logistico[-7]
cov <- paste(var_logistico, collapse = " + ")
form <- paste0("y ~ ",cov, collapse = " + ")
form_ger <- as.formula(paste(c(form, paste(c("(1", "Pin.code)"), collapse = "|")), collapse = " + "))
form_ger
err.ger <- rep(NA, K)

for(i in 1:K){
  print(i)
  M1 <- glmer(form_ger , data = dati[foldid != i,], family = "binomial", 
              control = glmerControl(optimizer="bobyqa", optCtrl = list(maxfun = 100000)))
  summary(M1)
  ger.pred.prob <- predict(M1, type = "response", newdata = dati[foldid == i, ], allow.new.levels = T)
  print(head(ger.pred.prob))
  indice <- which.min(sapply(probs, function(x) cz(modello = ger.pred.prob, soglia = x, y = y.num[foldid == i])))  
  ger.tab = table(ger.pred.prob > probs[indice], y.num[foldid == i])
  print(ger.tab)
  err.ger[i] = 1 - sum(diag(ger.tab))/sum(ger.tab)
  
}


err.ger
ger.e <- mean(err.ger)
ger.sd <- sd(err.ger)
cat("CV Error: ", ger.e," (sd:", ger.sd,")\n")
accuracy.matrix <- rbind(accuracy.matrix, c(1-ger.e, ger.sd))
rownames(accuracy.matrix)[nrow(accuracy.matrix)] = c("Gerarchico")
accuracy.matrix

M1 <- glmer(form_ger , data = dati2, family = "binomial")
summary(M1)
ranef(M1)
effetti.casuali.gaia <- round(ranef(M1)$Pin.code,2)
save(effetti.casuali.gaia, file = "ranefGaia.RData")
# M1 <- glmer(form_ger , data = dati, family = "binomial", 
#             control = glmerControl(optimizer="bobyqa", optCtrl = list(maxfun = 100000)))
# summary(M1)
# ranef(M1)
# 
# 
# 
# M2 <- glmer(form_ger , data = dati, family = "binomial", 
#             control = glmerControl(optimizer="bobyqa", optCtrl = list(maxfun = 100000)))
# summary(M2)
# ranef(M2)
# round(ranef(M2)$Pin.code,5)
