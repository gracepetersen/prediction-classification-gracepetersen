# SVM Polynomial

# Load packages
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)
library(kernlab)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")

# Set up parallel processing -----
## Unix and macOS only
registerDoMC(cores = 8)


# USING RECIPE 1 ######################################################

# Set up poly model
svm_poly_model <- svm_poly(
  mode = "classification",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>%
  set_engine("kernlab")

# Params
svm_poly_params <- parameters(svm_poly_model)

# Create grid 
svm_poly_grid <- grid_regular(svm_poly_params, levels = 5)

load("results/recipe1")

# Set up workflow
svm_poly_workflow1 <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(recipe1)


# Tune model
svm_poly_tuned1 <- tune_grid(svm_poly_workflow1, 
                            folds,
                            grid = svm_poly_grid,
                            control = control_grid(save_pred = TRUE, # Create an extra column for each prediction
                                                   save_workflow = TRUE, # Lets you use extract_workflow
                                                   verbose = TRUE,
                                                   parallel_over = "everything")) 



#save as rda
save(svm_poly_tuned1,
     file = "results/svm_poly_tuned1.rda")

load("results/svm_poly_tuned1.rda")

# RMSE 10.1
svm_poly_tuned1 %>% 
  show_best()

# Using Recipe 3 #############################################

load("results/recipe3_lasso")

# Set up poly model
svm_poly_model <- svm_poly(
  mode = "classification",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>%
  set_engine("kernlab")

# Params
svm_poly_params <- parameters(svm_poly_model)

# Create grid 
svm_poly_grid <- grid_regular(svm_poly_params, levels = 5)

# Set up workflow
svm_poly_workflow3 <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(recipe3_lasso)


# Tune model
svm_poly_tuned3 <- tune_grid(svm_poly_workflow3, 
                             folds,
                             grid = svm_poly_grid,
                             control = control_grid(save_pred = TRUE, # Create an extra column for each prediction
                                                    save_workflow = TRUE, # Lets you use extract_workflow
                                                    verbose = TRUE,
                                                    parallel_over = "everything")) 



#save as rda
save(svm_poly_tuned3,
     file = "results/svm_poly_tuned3.rda")

load("results/svm_poly_tuned1.rda")

# RMSE 10.1
svm_poly_tuned1 %>% 
  show_best()