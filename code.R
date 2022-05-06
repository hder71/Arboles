library(tidyverse)
library(tidymodels)

credit <- read.csv("/Users/hectorestrada/Downloads/creditcard.csv")

credit$Time <- NULL

credit %>% filter(Class==0) %>% count(Class)
prop.table(table(credit$Class)) #proporción en porcentaje 
#492 false transactions
#284315 true transactions

library(ggplot2)
library(DataExplorer)


ggplot(credit, aes(Class, fill = Amount)) + geom_bar()
#####

ggplot(credit, aes(x = factor(Class), y = Amount)) + geom_boxplot() + 
  labs(x = 'Class', y = 'Amount') +
  ggtitle("Distribución de clases por gasto") 

credit %>% group_by(Class) %>% summarise(mean(Amount), median(Amount))
#La media de las transacciones fraudulentas es más alta comparada
#a las transacciones correctas, esto puede deberse principalmente 
#a la desviación de los datos 
#####
introduce(credit)
plot_intro(credit)
#En el Data Set nuestras columnas son continuas, no tenemos datos faltantes dentro de las columnas
######
plot_correlation(na.omit(credit), type = "c")
#Existe una buena correlación entre el monto y la clase

###########
data<-data.frame(credit)
normalize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))}
data$Amount <- normalize(data$Amount)

#t-SNE: La visualización nos permite encontrar patrones en los datos de una manera 
#gráfica, si no existe algún patrón esto puede complicar el modelado de 
#la información 
install.packages("Rtsne")
library(Rtsne)
# Use 20% of data to compute t-SNE
tsne_subset <- 1:as.integer(0.2*nrow(data))
tsne <- Rtsne(data[tsne_subset,-c(1, 31)], perplexity = 20, 
              theta = 0.5, pca = F, verbose = T, max_iter = 500, check_duplicates = F)
classes <- as.factor(data$Class[tsne_subset])
tsne_mat <- as.data.frame(tsne$Y)
ggplot(tsne_mat, aes(x = V1, y = V2)) + geom_point(aes(color = classes)) + theme_minimal() +  ggtitle("t-SNE visualisation of transactions") + scale_color_manual(values = c("#E69F00", "#56B4E9"))
#Podemos encontrar que la mayoria de los casos fraudulentos 
#se encuentran en una sección del mapa
#######
set.seed(123)
credit_split <- credit %>% initial_split(strata = Class)
train <- training(credit_split)
test <- testing(credit_split)

set.seed(555)

train1 <- bootstraps(train, strata = Class,  times= 2000)
train1 <- train1$splits[[1]]
train1 <- as.data.frame(train1)
#add to train 
library(h2o)

# To launch the H2O cluster

localH2O <- h2o.init(nthreads = -1)
h2o.init()

h2o.removeAll()

datos_h2o <- as.h2o(x = train1, destination_frame = "datos_h2o")

particiones     <- h2o.splitFrame(data = datos_h2o, ratios = c(0.6,0.2), seed = 1234)
datos_train_h2o <- h2o.assign(data = particiones[[1]], key = "datos_train_H2O")
datos_valid_h2o <- h2o.assign(data = particiones[[2]], key = "datos_valid_H2O")
datos_test_h2o  <- h2o.assign(data = particiones[[3]], key = "datos_test_H2O")

#ARBOLES DE LA BARRANCA

random_forest_model <- h2o.randomForest(
  training_frame = datos_train_h2o, 
  validation_frame = datos_valid_h2o, 
  x = 1:29, 
  y = 30,    
  model_id = "rf",  
  ntrees = 200, 
  stopping_rounds = 2, 
  score_each_iteration = T, 
  seed = 1000000  
)
summary(random_forest_model)

c1 <- datos_h2o[,30]


gbm_model <- h2o.gbm(
  training_frame = datos_train_h2o, 
  validation_frame = datos_valid_h2o, 
  x = 1:29, 
  y = 30,    
  model_id = "gbm", 
  seed = 2000000   
) 

summary(gbm_model)

scoring <- as.data.frame(gbm_model@model$scoring_history)
importancia <- as.data.frame(gbm_model@model$variable_importances)

ggplot(data = importancia,
       aes(x = reorder(variable, scaled_importance), y = scaled_importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Importancia de los predictores en el modelo GBM",
       subtitle = "Importancia en base a la reducci?n del error cuadr?tico medio",
       x = "Predictor",
       y = "Importancia relativa") +
  theme_bw()

#Aqui esta de la otra forma.
train.h2o <- as.h2o(train1)
test.h2o <- as.h2o(test)

#dependent variable 
y.dep <- 30

#independent variables 
x.indep <- c(1:29)


# Logistic Regression in H2O

logistic.model <- h2o.glm( x = x.indep, y = y.dep, training_frame = train.h2o, family = "binomial")
h2o.performance(logistic.model)

#make predictions
predict.reg <- as.data.frame(h2o.predict(logistic.model, test.h2o))
h2o.performance(logistic.model,test.h2o)
 table(predict.reg[1])
 
 
#random forest part
 
system.time(
  rforest.model <- h2o.randomForest(y=31, x=x.indep, training_frame = train.h2o, ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122)
)
 
h2o.performance(rforest.model)
 
 #check variable importance
h2o.varimp(rforest.model)

#GBM

system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122)
)

h2o.shutdown(prompt = TRUE)


