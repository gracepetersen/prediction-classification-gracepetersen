# SVM radial

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)

## Unix and macOS only
registerDoMC(cores = 8)

# Seed
set.seed(3013)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")
load("results/recipe1")

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

# Set up workflow
svm_radial_workflow <- workflow() %>% 
  add_model(svm_radial_model) %>% 
  add_recipe(recipe1)

ctrl_grid <- control_stack_grid()

# Tuning/fitting ----
svm_radial_res <- svm_radial_workflow %>%
  tune_grid(
    resamples = folds,
    grid = svm_radial_grid,
    control = ctrl_grid
  )

# Write out results & workflow
save(svm_radial_res, file = "model_info/svm_radial_res.rda")
