# SVM radial


# Load packages
library(tidyverse)
library(tidymodels)
library(doMC)
library(kernlab)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")

# Set up parallel processing -----
## Unix and macOS only
registerDoMC(cores = 8)

# Set up radial model
svm_radial_model <- svm_rbf(
  mode = "classification", 
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

# Params
svm_radial_params <- parameters(svm_radial_model)

# Create grid 
svm_radial_grid <- grid_regular(svm_radial_params, levels = 5)

load("results/recipe1")
# Set up workflow
svm_radial_workflow1 <- workflow() %>% 
  add_model(svm_radial_model) %>% 
  add_recipe(recipe1)


# Tune model
svm_radial_tuned1 <- tune_grid(svm_radial_workflow1, 
                              folds,
                              grid = svm_radial_grid,
                              control = control_grid(save_pred = TRUE, # Create an extra column for each prediction
                                                     save_workflow = TRUE, # Lets you use extract_workflow
                                                     verbose = TRUE,
                                                     parallel_over = "everything")) 

#save as rda
save(svm_radial_tuned1,
     file = "results/svm_radial_tuned1.rda")

load("results/svm_radial_tuned1.rda")

# .557 best
svm_radial_tuned1 %>% 
  show_best()

