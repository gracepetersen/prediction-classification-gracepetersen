# Elastic Net

# Elastic Net Tuning

# Load packages
library(tidyverse) 
library(tidymodels)
library(doMC)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")

# Set up parallel processing -----
## Unix and macOS only
registerDoMC(cores = 8)

# Define model engine -----
en_model <- logistic_reg(mode = "classification",
                         penalty = tune(), 
                         mixture = tune()) %>% 
  set_engine("glmnet")

# Params 
en_params <- extract_parameter_set_dials(en_model)

# Create grid  -----
en_grid <- grid_regular(en_params, levels = 5)

load("results/recipe1")

# update recipe
recipe1_interact <- recipe1 %>% 
  step_interact(~all_numeric_predictors()^2)

# Set up workflow -----
en_workflow1 <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(recipe1_interact)

# Tune parameters -----
en_tuned1 <- tune_grid(en_workflow1, 
                      folds,
                      grid = en_grid,
                      control = control_grid(save_pred = TRUE, 
                                             save_workflow = TRUE,
                                             verbose = TRUE,
                                             parallel_over = "everything"))

#save as rda
save(en_tuned1,
     file = "results/en_tuned1.rda")

load("results/en_tuned1.rda")

# .553
en_tuned1 %>% 
  show_best()

en_workflow1_tuned <- en_workflow1 %>% 
  finalize_workflow(select_best(en_tuned1, metric = "roc_auc"))

# Fit entire training data set to workflow
en1_results <- fit(en_workflow1_tuned, training)

en1_pred <- testing %>% 
  bind_cols(predict(en1_results, testing)) %>% 
  select(id, .pred_class) %>% 
  rename(y = .pred_class)

write_csv(en1_pred, file = "submissions/en1_pred.csv")