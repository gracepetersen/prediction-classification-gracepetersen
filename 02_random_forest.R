# Random Forest

# Load packages
library(tidyverse)
library(tidymodels)
library(doMC)

# Seed
set.seed(3013)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")

# Set up parallel processing -----
## Unix and macOS only
registerDoMC(cores = 8)

# Write model
rf_model <- rand_forest(mode = "classification",
                        min_n = tune(),
                        mtry = tune(),
                        trees = 100) %>% 
  set_engine("ranger", importance = "impurity")

# Params 
rf_params <- parameters(rf_model) %>% 
  recipes::update(mtry = mtry(range = c(5, 10)))

# Create grid 
rf_grid <- grid_regular(rf_params, levels = 5)

############################################################################
# Recipe 1 ################################################################
# load("results/recipe1")
# 
# rf_workflow1 <- workflow() %>% 
#   add_model(rf_model) %>% 
#   add_recipe(recipe1)
# 
# # Random forest
# registerDoMC(cores = 8)
# rf_tuned1 <- tune_grid(rf_workflow1, 
#                       folds,
#                       grid = rf_grid,
#                       control = control_grid(save_pred = TRUE, 
#                                              save_workflow = TRUE,
#                                              verbose = TRUE,
#                                              parallel_over = "everything"))
# 
# save(rf_tuned1,
#      file = "results/rf_tuned1.rda")
# 
# load('results/rf_tuned1.rda')
# 
# # .577
# rf_tuned1 %>% 
#   show_best()
# 
# rf_workflow_tuned <- rf_workflow1 %>% 
#   finalize_workflow(select_best(rf_tuned1, metric = "roc_auc"))
# 
# # Fit entire training data set to workflow
# rf1_results <- fit(rf_workflow_tuned, training)
# 
# # obtain probability of category
# rf_prob1 <- predict(rf1_results, testing)
# 
# # bind cols together
# rf_results_prob <- testing %>% 
#   select(id) %>%
#   bind_cols(rf_prob1) %>% 
#   select(id, .pred_class) %>% 
#   rename(y = .pred_class)
# 
# write_csv(rf_results_prob, file = "submissions/rf1_prob.csv")
# 
# rf1_pred <- testing %>% 
#   bind_cols(predict(rf1_results, testing)) %>% 
#   select(id, .pred_class) %>% 
#   rename(y = .pred_class)
# 
# write_csv(rf1_pred, file = "submissions/rf1_pred.csv")


############################################################################
# Recipe 2 ################################################################
load("results/recipe2")

rf_workflow2 <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe2)

# Random forest
registerDoMC(cores = 8)
rf_tuned2 <- tune_grid(rf_workflow2, 
                       folds,
                       grid = rf_grid,
                       control = control_grid(save_pred = TRUE, 
                                              save_workflow = TRUE,
                                              verbose = TRUE,
                                              parallel_over = "everything"))

save(rf_tuned2,
     file = "results/rf_tuned2.rda")

load('results/rf_tuned2.rda')

# .577
rf_tuned2 %>% 
  show_best()

rf_workflow_tuned2 <- rf_workflow2 %>% 
  finalize_workflow(select_best(rf_tuned2, metric = "roc_auc"))

# Fit entire training data set to workflow
rf2_results <- fit(rf_workflow_tuned2, training)

# obtain probability of category
rf_prob2 <- predict(rf2_results, testing, type = "prob")

# bind cols together
rf_results_prob2 <- testing %>% 
  select(id) %>%
  bind_cols(rf_prob2) %>% 
  select(id, .pred_1) %>% 
  rename(y = .pred_1)

write_csv(rf_results_prob2, file = "submissions/rf2_prob.csv")

rf2_pred <- testing %>% 
  bind_cols(predict(rf2_results, testing)) %>% 
  select(id, .pred_class) %>% 
  rename(y = .pred_class)

write_csv(rf2_pred, file = "submissions/rf2_pred.csv")

############################################################################
# Recipe 3 lasso ################################################################
load("results/recipe3_lasso")

rf_workflow3 <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe3_lasso)

# Random forest
registerDoMC(cores = 8)
rf_tuned3 <- tune_grid(rf_workflow3, 
                       folds,
                       grid = rf_grid,
                       control = control_grid(save_pred = TRUE, 
                                              save_workflow = TRUE,
                                              verbose = TRUE,
                                              parallel_over = "everything"))

save(rf_tuned3,
     file = "results/rf_tuned3.rda")

load('results/rf_tuned3.rda')

rf_table <- rf_tuned3 %>% 
  show_best()

save(rf_table, file = "results/rf_table")

rf_workflow_tuned3 <- rf_workflow3 %>% 
  finalize_workflow(select_best(rf_tuned3, metric = "roc_auc"))

# Fit entire training data set to workflow
rf3_results <- fit(rf_workflow_tuned3, training)

# obtain probability of category
rf_prob3 <- predict(rf3_results, testing, type = "prob")

# bind cols together
rf_results_prob3 <- testing %>% 
  select(id) %>%
  bind_cols(rf_prob3) %>% 
  select(id, .pred_1) %>% 
  rename(y = .pred_1)

#### BEST OUTCOME #########################
write_csv(rf_results_prob3, file = "submissions/rf3_prob.csv")

rf3_pred <- testing %>% 
  bind_cols(predict(rf3_results, testing)) %>% 
  select(id, .pred_class) %>% 
  rename(y = .pred_class)

write_csv(rf3_pred, file = "submissions/rf3_pred.csv")