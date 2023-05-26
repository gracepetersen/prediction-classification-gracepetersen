# Ensemble Model


# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load("model_info/rf_res.rda")
load("model_info/bt_res2.rda")

# Load in tuning data
load('results/tuning_setup')
load("results/recipe2")

# Create data stack ----
data_st <- 
  stacks() %>%
  add_candidates(rf_res) %>%
  add_candidates(bt_res2)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(3013)

model_st <-
  data_st %>%
  blend_predictions(penalty = blend_penalty)

# fit to ensemble to entire training set ----
fitted_model_st <-
  model_st %>%
  fit_members()

# Explore and assess trained ensemble model
assess <- testing %>%
  bind_cols(predict(fitted_model_st, .))

# obtain probability of category
ensemble_prob2 <- predict(fitted_model_st, testing, type = "prob")

# bind cols together
ensemble_prob2 <- testing %>% 
  select(id) %>% 
  select(id, .pred_1) %>% 
  rename(y = .pred_1)

assess_table <- assess %>% 
  select(id, .pred_class) %>% 
  rename(y = .pred_class)

write_csv(assess_table, file = "submissions/ensemble_model_SUB2.csv")
