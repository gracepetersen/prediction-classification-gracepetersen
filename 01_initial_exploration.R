# Initial Exploration

######################################################################
# Load package(s) ####################################################
library(tidymodels)
library(tidyverse)
library(doMC)
library(knitr)
library(caret)
library(randomForest)
library(varImp)


# handle common conflicts
tidymodels_prefer()

## Unix and macOS only
registerDoMC(cores = 8)

# Seed
set.seed(3013)


######################################################################
# Load Data ##########################################################

training <- read_csv("data/train.csv")
testing <- read_csv("data/test.csv")

training <- training %>% 
  mutate(y = as.factor(y))

my_split <- initial_split(training, prop = 0.7, strata = y)
test_data <- testing(my_split)
train_data <- training(my_split)

#######################################################################
# Distribution of Y ###################################################

ggplot(train_data, aes(x = y)) +
  geom_bar()

#######################################################################
# Check for missingness ###############################################
missing_list <- list()

for(var in colnames(train_data)){
  missing_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    filter(is.na(!!sym(var))) %>% 
    summarise(num_missing = n())
}

missing_tbl <- enframe(unlist(missing_list)) %>% 
  mutate(pct = value/3849) %>% 
  arrange(desc(pct)) %>% 
  filter(pct >= 0.05)

missing_vars <- missing_tbl %>% 
  pull(name)

train_data <- train_data %>% 
  select(!all_of(missing_vars))

######################################################################
# Check for 0 variance ###############################################

# Use step_zv()

var_list <- list()

for(var in colnames(train_data)){
  var_list[var] <- train_data %>% 
    select(any_of(var)) %>%
    summarise(sd = sd(!!sym(var), na.rm = TRUE))
}

zv_tbl <- enframe(unlist(var_list)) %>% 
  arrange(value)

# Remove zero variance
# High variance might benefit from a transformation
zero_var <- zv_tbl %>% 
  filter(value == 0) %>% 
  pull(name) #data$var

# Update training data to remove unwanted variables
train_data <- train_data %>% 
  select(!all_of(zero_var))


######################################################################
# Remove highly correlated vars ######################################

# Remove variables with a 0.99 correlation
tmp <- cor(train_data)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0

train_data <- 
  train_data[, !apply(tmp, 2, function(x) any(abs(x) > 0.8, na.rm = TRUE))]


######################################################################
# Remove Near Zero Variance ##########################################

train_data <- train_data[, -nearZeroVar(train_data)]

######################################################################
# Find 30 most important vars ########################################

train_data_na_rm <- train_data %>% 
  na.omit()

# Train a Random Forest classifier
rf_model_varImp <- randomForest(y ~ ., data = train_data_na_rm, ntree = 500, importance = TRUE)

# Print variable importance scores
var_imp <- importance(rf_model_varImp)

save(var_imp, file = "results/var_imp")

load("results/var_imp")

var_imp <- var_imp[order(var_imp[, 1], decreasing = TRUE), ]

var_imp %>% 
  head(30) %>% 
  row.names()


######################################################################
# Use V-fold cross validation ########################################

folds <- vfold_cv(training, v = 5, repeats = 3,
                 strata = y)

save(training, testing, folds, file = "results/tuning_setup")

######################################################################
# Recipe 1 ###########################################################

recipe1 <- recipe(y ~ x727 + x155 + x165 + x122 + x511 + x746 + x715 + x582 + x371 + x262 + x600 + x514 + x312 + x187 + x195 + x278 + x088 + x208 + x703 + x095 + x423 + x753 + x614 + x735 + x705 + x507 + x625 + x695 + x684 + x633,
                  data = training) %>% 
  step_impute_mean(all_numeric_predictors()) %>% 
  step_impute_mode(all_factor_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  # Remove variables with zero variance
  step_nzv(all_predictors()) %>% 
  # Center and scale all predictors
  step_normalize(all_numeric_predictors())

save(recipe1, file = "results/recipe1")


recipe2 <- recipe(y ~ x727 + x155 + x165 + x122 + x511 + x746 + x715 + x582 + x371 + x262 + x600 + x514 + x312 + x187 + x195 + x278 + x088 + x208 + x703 + x095 + x423 + x753 + x614 + x735 + x705 + x507 + x625 + x695 + x684 + x633,
                  data = training) %>% 
  step_impute_mean(all_numeric_predictors()) %>% 
  step_impute_mode(all_factor_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  # Remove variables with zero variance
  step_nzv(all_predictors()) %>% 
  # Center and scale all predictors
  step_normalize(all_numeric_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors())

save(recipe2, file = "results/recipe2")


