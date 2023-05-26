# Boosted trees

# Boosted Trees

# Load packages
library(tidyverse)
library(tidymodels)
library(doMC)

# Handle common conflicts
tidymodels_prefer()

# Load in tuning data
load("results/tuning_setup")
load("results/recipe1")

# Set up parallel processing -----
## Unix and macOS only
registerDoMC(cores = 8)

# Create model
boost_model <- boost_tree(mode = "classification",
                          min_n = tune(),
                          mtry = tune(),
                          learn_rate = tune()) %>% 
  set_engine("xgboost", importance = "impurity")

# Params
boost_params <- parameters(boost_model) %>% 
  update(mtry = mtry(range = c(1,10))) %>% 
  update(learn_rate = learn_rate(range = c(-5, -0.2)))

# Create grid 
boost_grid <- grid_regular(boost_params, levels = 5)

# Random forest
boost_workflow1 <- workflow() %>% 
  add_model(boost_model) %>% 
  add_recipe(recipe1)


# Random forest
boost_tuned1 <- tune_grid(boost_workflow1, 
                          folds,
                          grid = boost_grid,
                          control = control_grid(save_pred = TRUE, # Create an extra column for each prediction
                                                 save_workflow = TRUE, # Lets you use extract_workflow
                                                 verbose = TRUE,
                                                 parallel_over = "everything")) 


#save as rda
save(boost_tuned1,
     file = "results/bt_tuned1.rda")

load("results/bt_tuned1.rda")

# .578
boost_tuned1 %>% 
  show_best()

bt_workflow_tuned1 <- boost_workflow1 %>% 
  finalize_workflow(select_best(boost_tuned1, metric = "roc_auc"))

# Fit entire training data set to workflow
bt1_results <- fit(bt_workflow_tuned1, training)


# obtain probability of category
bt_prob1 <- predict(bt1_results, testing, type = "prob")

# bind cols together
bt_results_prob <- testing %>% 
  select(id) %>%
  bind_cols(bt_prob1) %>% 
  select(id, .pred_1) %>% 
  rename(y = .pred_1)

write_csv(bt_results_prob, file = "submissions/bt1_prob.csv")






bt1_pred <- testing %>% 
  bind_cols(predict(bt1_results, testing)) %>% 
  select(id, .pred_class) %>% 
  rename(y = .pred_class)

write_csv(bt1_pred, file = "submissions/bt1_pred.csv")