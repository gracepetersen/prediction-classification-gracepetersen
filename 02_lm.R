# Random Forest

# Load packages
library(tidyverse)
library(tidymodels)
library(doMC)

# Load in tuning data
load("results/tuning_setup")
load("results/recipe1")

# Seed
set.seed(3013)

# Handle common conflicts
tidymodels_prefer()

lm_spec <- linear_reg() %>% 
  set_engine("lm")

lm_workflow <- workflow() %>% 
  add_model(lm_spec) %>% 
  add_recipe(recipe1)

keep_pred <- control_resamples(save_pred = TRUE)

lm_fit_folds <- fit_resamples(lm_workflow, 
                              resamples = folds,
                              control = keep_pred)

write_rds(lm_fit_folds, file = "results/lm_fit_folds.rds")