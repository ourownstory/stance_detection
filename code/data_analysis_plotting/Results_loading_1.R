### Results loading
library(tidyverse)
library(stringr)
library(forcats)
# help(package = 'forcats')


###Load data function#################################################################
## Paths
path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/'


## Load data function
load_data <- function(names) {
  num <-  length(names)
  data <- read_csv(str_c(path_res, names[1]))
  if (num > 1) {
    for (i in 2:num) {
      data <- data %>% 
        bind_rows(read_csv(str_c(path_res, names[i])))
    }
  }
  return(data)
}

## Load data combine function
load_data_combine <- 
  function(perf_names, loss_names, model_ = NULL, xp_ = NULL) {
    perf <- load_data(perf_names) %>% 
      mutate(epoch = epoch + 1)
    loss <- load_data(loss_names) %>% 
      rename(train_loss = loss)
    data <- 
      inner_join(perf, loss) %>% 
      mutate(
        model = model_,
        xp = xp_,
        downsample = FALSE) %>% 
      select(model, xp, downsample, everything())
    return(data)
  }


#####LSTM Experiment 01###########################################################################
## Experiment 01
path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/'

# Sensitivity Analysis over max_length and n_layers, without downsampling. 
# Naive data splitting testing script: test_script3_config.py 
# 
# Sensitivity Analysis over max_length and n_layers, with downsampling. 
# Downsample first and then split data. Naive data splitting wrt headlines. 
# testing script: test_script4_config.py 


## Data 01

# {r 1 No Downsampling max_length}
### No Downsampling
## Collect max_length XP data
#max_length
perf_names <- 
  str_c('old(lstm)/', 
        c(
          'perf_148979940586.csv',
          'perf_148980307237.csv',
          'perf_148980957009.csv'
        )
  )
#max_length
loss_names <- 
  str_c('old(lstm)/', 
        c(
          'losses_148979940587.csv',
          'losses_148980307238.csv',
          'losses_148980957009.csv'
        )
  )

## Read data #max_length
perf <- load_data(perf_names) %>% 
  mutate(epoch = epoch + 1)
loss <- load_data(loss_names) %>% 
  rename(train_loss = loss)

results1 <- 
  inner_join(perf, loss) %>% 
  mutate(xp = 'max_length',
         downsample = FALSE) %>% 
  select(xp, downsample, everything())

# unique(perf$max_length)
# unique(loss$max_length)




# {r 2 No Downsampling n_layers}
### No Downsampling
## Collect n_layers data
#n_layers
perf_names <- 
  str_c('old(lstm)/', 
        c(
          'perf_148982227549.csv', 
          'perf_148981534989.csv', 
          'perf_148981166478.csv'
        )
  )
#n_layers
loss_names <- 
  str_c('old(lstm)/', 
        c(
          'losses_148973251886.csv', 
          'losses_148973620163.csv', 
          'losses_148974313682.csv'
        )
  )

## Read data #n_layers
perf <- load_data(perf_names) %>% 
  mutate(epoch = epoch + 1)
loss <- load_data(loss_names) %>% 
  rename(train_loss = loss)

results2 <- 
  inner_join(perf, loss) %>% 
  mutate(xp = 'n_layers',
         downsample = FALSE) %>% 
  select(xp, downsample, everything())

# unique(perf$max_length)
# unique(loss$max_length)




# {r 3 With Downsampling max_length}
### With Downsampling
## Collect max_length XP data
#max_length
perf_names <- 
  str_c('old(lstm)/', 
        c(
          'perf_148978975462.csv', 
          'perf_148978680439.csv', 
          'perf_148978512014.csv'
        )
  )
#max_length
loss_names <- 
  str_c('old(lstm)/', 
        c(
          'losses_148978975462.csv', 
          'losses_148978680439.csv', 
          'losses_148978512015.csv'
        )
  )

## Read data #max_length
perf <- load_data(perf_names) %>% 
  mutate(epoch = epoch + 1)
loss <- load_data(loss_names) %>% 
  rename(train_loss = loss)

results3 <- 
  inner_join(perf, loss) %>% 
  mutate(xp = 'max_length',
         downsample = TRUE) %>% 
  select(xp, downsample, everything())

# unique(perf$max_length)
# unique(loss$max_length)

# results3 <- 
#   anti_join(perf, loss)




# {r 4 With Downsampling n_layers}
### With Downsampling
## Collect n_layers data
#n_layers
perf_names <- 
  str_c('old(lstm)/', 
        c(
          'perf_148979553502.csv', 
          'perf_148979239173.csv', 
          'perf_148979071576.csv'
        )
  )
#n_layers
loss_names <- 
  str_c('old(lstm)/', 
        c(
          'losses_148979071576.csv', 
          'losses_148979239173.csv', 
          'losses_148979553502.csv'
        )
  )

## Read data #n_layers
perf <- load_data(perf_names) %>% 
  mutate(epoch = epoch + 1)
loss <- load_data(loss_names) %>% 
  rename(train_loss = loss)

results4 <- 
  inner_join(perf, loss) %>% 
  mutate(xp = 'n_layers',
         downsample = TRUE) %>% 
  select(xp, downsample, everything())

# unique(perf$max_length)
# unique(loss$max_length)




##Check all
# print('1 No Downsampling max_length') 
# sapply(results1 %>% select(1:12), unique)
# print('2 No Downsampling n_layers') 
# sapply(results2 %>% select(1:12), unique)
# print('3 With Downsampling max_length') 
# sapply(results3 %>% select(1:12), unique)
# print('4 With Downsampling n_layers') 
# sapply(results4 %>% select(1:12), unique)

##Combine all
results_lstm1 <- bind_rows(results1, results2, results3, results4)

# results_lstm1 %>% write_rds(str_c(path_res, 'old(lstm)/', 'results_lstm1.rds'))


#####LSTM Experiment 02###########################################################################
### Experiment 02
path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/basiclstm/wrangled/'

perf_names_drop <- 
  c(
  'perf_148991935973_drop.csv',
  'perf_148992159916_drop.csv',
  'perf_148992383928_drop.csv'#,
  # 'perf_148992558606_maxl.csv'
  )

perf_names_maxl <- 
  c(
    # 'perf_148991935973_drop.csv',
    # 'perf_148992159916_drop.csv',
    # 'perf_148992383928_drop.csv'#,
    'perf_148992558606_maxl.csv'
  )

loss_names_drop <- 
  c(
    'losses_148991935973_drop.csv',
    'losses_148992159916_drop.csv', 
    'losses_148992383928_drop.csv'#, 
    # 'losses_148992558606_maxl.csv',
    # 'losses_148992694411_maxl.csv'
  )

loss_names_maxl <- 
  c(
    # 'losses_148991935973_drop.csv',
    # 'losses_148992159916_drop.csv', 
    # 'losses_148992383928_drop.csv', 
    'losses_148992558606_maxl.csv',
    'losses_148992694411_maxl.csv'
  )

## Read data #max_length, dropout
perf_drop <- load_data(perf_names_drop) %>% 
  mutate(epoch = epoch + 1) %>% 
  mutate(xp = 'dropout')
perf_maxl <- load_data(perf_names_maxl) %>% 
  mutate(epoch = epoch + 1) %>% 
  mutate(xp = 'max_length')
# perf <- bind_rows(perf_drop, 

loss_drop <- load_data(loss_names_drop) %>% 
  rename(train_loss = loss) 

loss_maxl <- load_data(loss_names_maxl) %>% 
  rename(train_loss = loss) 

results_drop <- 
  inner_join(perf_drop, loss_drop) %>% 
  mutate(#xp = 'max_length',
         downsample = FALSE) %>% 
  select(xp, downsample, everything())

results_maxl <- 
  inner_join(perf_maxl, loss_maxl) %>% 
  mutate(#xp = 'max_length',
    downsample = FALSE) %>% 
  select(xp, downsample, everything())


## add baselines for other experiments
# base1 <- 
#   results_lstm1 %>% 
#   filter(xp == 'max_length') %>% 
#   filter(max_length == 150) %>% 
#   mutate(xp = 'base_150')

base_drop <- 
  results_lstm1 %>% 
  filter(xp == 'max_length') %>% 
  filter(max_length == 75) %>% 
  mutate(xp = 'dropout')

### final results of Exp 2
results_lstm2 <- 
  bind_rows(results_drop, results_maxl, base_drop)

# unique(perf$max_length)
# unique(loss$max_length)

#####Combine LSTM data#################################################################
## Combine LSTM data


results_lstm <- bind_rows(results_lstm1, results_lstm2) %>% 
  mutate(model = 'Basic LSTM') %>% 
  select(model, xp, everything())

# results_lstm %>% write_rds(str_c(path_res, 'results_lstm.rds'))

#####Attention_data 01###########################################################################
## Attention_data

## Paths
path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/attention/wrangled/'

### Experiments
perf_names <- c(
'perf_148990711237_base.csv'
)
loss_names <- c(
'losses_148990711238_base.csv'
)
att_base <- load_data_combine(perf_names, loss_names,
                                      model_ = 'attention', xp_ = 'base150')

perf_names <- c(
'perf_148990459067_maxl.csv',
'perf_148990711237_maxl.csv',
'perf_148991552302_maxl.csv',
'perf_148992831854_maxl.csv'
)
loss_names <- c(
'losses_148990459067_maxl.csv',
'losses_148990711238_maxl.csv',
'losses_148991552302_maxl.csv',
'losses_148992831855_maxl.csv'
)
att_maxl <- load_data_combine(perf_names, loss_names, 
                                      model_ = 'attention', xp_ = 'max_length')

perf_names <- c(
'perf_148990711237_att.csv',
'perf_148993077478_att.csv',
'perf_148993325155_att.csv',
'perf_148989809876_att.csv'
)
loss_names <- c(
'losses_148990711238_att.csv',
'losses_148993077478_att.csv',
'losses_148993325156_att.csv',
'losses_148989809876_att.csv'
)
att_att <- load_data_combine(perf_names, loss_names, 
                                      model_ = 'attention', xp_ = 'attention_length')

perf_names <- c(
'perf_148990711237_lr.csv',
'perf_148993571229_lr.csv',
'perf_148993821388_lr.csv'
)
loss_names <- c(
'losses_148990711238_lr.csv',
'losses_148993571229_lr.csv',
'losses_148993821388_lr.csv'
)
att_lr <- load_data_combine(perf_names, loss_names, 
                                     model_ = 'attention', xp_ = 'lr')

### att_att has 40 rows too much!! because: max(att_att$n_epochs) is 50 ##
results_att1 <- bind_rows(#att_base, 
                         att_maxl, att_att, att_lr)

# results_att %>% write_rds(str_c(path_res,  'results_attention.rds'))


#####Attention_data 02, Combine###########################################################################
## Attention_data
path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/attention/wrangled/'
perf_names <- c(
'perf_148998721083_nlay.csv',
'perf_148997977122_nlay.csv'
)

results_att2 <- load_data(perf_names) %>% 
  mutate(epoch = epoch + 1) %>% 
  mutate(model = 'attention', xp = 'n_layers') %>% 
  mutate(downsample = FALSE)

results_att2 <- bind_rows(results_att2, (att_base %>% mutate(xp = 'n_layers')))

results_att <- bind_rows(results_att2, results_att1) %>% 
  mutate(model = 'Attention LSTM')
# results_att %>% write_rds(str_c(path_res,  'results_attention.rds'))


#####Conditional ###########################################################################
### Conditional Data

path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/conditional/wrangled/'
perf_names <- c(
'perf_148995686583_max75.csv',
'perf_148996029453_max150.csv',
'perf_148996505757_max300.csv'
)

results_cond1 <- load_data(perf_names) %>% 
  mutate(epoch = epoch + 1,
         max_length = b_max_len,
         xp = 'max_length') 


perf_names <- c(
'perf_148996029453_max150.csv',
'perf_149000293039_nlay.csv',
'perf_14899932587_nlay.csv'
)

results_cond2 <- load_data(perf_names) %>% 
  mutate(epoch = epoch + 1,
         max_length = b_max_len,
         xp = 'n_layers') 

results_cond <- bind_rows(results_cond1, results_cond2) %>% 
  mutate(model = 'CEA LSTM')


# results_cond %>% write_rds(str_c('C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/', 
#                                  'results_cond.rds'))


#####BOW Data####################################################################

path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/bow/wrangled/'

perf_names <- c(
'perf_14899707540.csv',
'perf_148996933874.csv',
'perf_148996941198.csv',
'perf_148996948691.csv',
'perf_148997030832.csv',
'perf_148997038112.csv',
'perf_148997045544.csv',
'perf_148997052838.csv',
'perf_148997060346.csv',
'perf_148997067842.csv',
'perf_148997083262.csv',
'perf_148997091668.csv',
'perf_148997099947.csv',
'perf_148997108384.csv',
'perf_148997116878.csv'
)

results_bow <- load_data(perf_names) 

## add missing variable embedding 4 layer runs for 75, 300 and 600 max_length
results_bow_add <- results_bow %>% 
  filter(trainable_embeddings == 'Constant',
         n_layers == 3,
         b_max_len %in% c(75, 300, 600)) %>% 
  mutate(trainable_embeddings = 'Variable')
results_bow <- bind_rows(results_bow, results_bow_add)

##  add 150 max_len as n_layers experiment
results_bow_add <- results_bow %>% 
  filter(trainable_embeddings == 'Variable',
         b_max_len %in% c(150)) %>% 
  mutate(xp = 'n_layers')
results_bow <- bind_rows(results_bow, results_bow_add)

## add 4 layers as max_len experiment
results_bow_add <- results_bow %>% 
  filter(trainable_embeddings == 'Variable',
         n_layers == 3,
         xp == 'layers') %>% 
  mutate(xp = 'max_length')
results_bow <- bind_rows(results_bow, results_bow_add)



results_bow <- results_bow %>% 
  mutate(
    epoch = epoch + 1,
    n_layers = n_layers + 1,
    max_length = b_max_len,
    model = 'BOW'
    )

# results_bow %>% write_rds(str_c(path_res, 'results_bow.rds'))

# model_bow <- results_bow %>% select(attention_length:xp) %>% distinct %>%
#   arrange(model, xp, trainable_embeddings, max_length, n_layers)

#####COMBINE ALL###########################################################################
path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/'

results <- bind_rows(
  results_lstm, 
  results_att %>% mutate(b_max_len = NA_integer_, h_max_len = NA_integer_), 
  results_cond %>% mutate(downsample = FALSE), 
  results_bow %>% mutate(downsample = FALSE)
  )

# results %>% write_rds(str_c('C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/', 'results.rds'))

## debug
# names(results_lstm)
# names(results_att)
# names(results_cond)
# names(results_bow)

# results_lstm %>% select(b_max_len) %>% head(5)
# results_att %>% select(b_max_len) %>% head(5)
# results_cond %>% select(b_max_len) %>% head(5)
# results_bow %>% select(b_max_len) %>% head(5)

# results_lstm %>% select(downsample) %>% head(5)
# results_att %>% select(downsample) %>% head(5)
# results_cond %>% select(downsample) %>% head(5)
# results_bow %>% select(downsample) %>% head(5)

#####FINAL RESULTS#################################################################
path_res <- 'C:/Users/OurOwnStory/GitHub/altfactcheckers/xp/final/'

results_final_bow  <- load_data('perf_149004079896.csv') %>% 
  mutate(max_length = b_max_len,
         model = 'BOW')
results_final_lstm <- load_data('perf_149004484911.csv') %>% 
  mutate(model = 'Basic LSTM')
results_final_att  <- load_data('perf_149004809705.csv') %>% 
  mutate(model = 'Attention LSTM') 
results_final_cond <- load_data('perf_149005331987.csv') %>% 
  mutate(max_length = b_max_len,
         model = 'CEA LSTM')

results_final <- bind_rows(results_final_bow, 
                           results_final_lstm, 
                           results_final_att, 
                           results_final_cond) %>% 
  mutate(epoch = epoch + 1)

# results_final %>% distinct(model) 

# results_final %>% write_rds(str_c(path_res, 'results_final.rds'))







