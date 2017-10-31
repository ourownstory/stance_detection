#!/usr/bin/env python2
# -*- coding: utf-8 -*-

######
# Call all models with different hyperparameters
######

# standard libs
import numpy as np

# our code imports
from execute_bow_config import run_bow
from execute_lstm_config import run_lstm
from execute_lstm_attention import run_lstm_attention
from execute_lstm_conditional import run_lstm_conditional

### Parameter Overview:
class Config:
  """Holds model hyperparams and data information.
  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation. Use self.config.? instead of Config.?
  """
  ### Parameter Overview:
  ## For all models:
  # main params,
  n_epochs = 40
  lr = 0.001
  batch_size = 128 
  n_classes = 4 
  hidden_size = 100
  n_layers = 0
  xp = None
  model = None

  ## Determined at data loading:
  embed_size = None  # not passed to config - assigned in get_data
  vocab_size = None  # not passed to config - assigned in get_data
  pretrained_embeddings = []  # not passed to config   - assigned in get_data
  num_samples = None  # only indirectly passed to comfig, If defined, shortens the dataset, Otherwise determined at data loading,
  downsample = False

  ## LSTM specific:
  # main params 
  dropout  = 0.8  ## Attention: this is the keep_prob! # not assigned to BOW
  # extra_hidden_size = None
  trainable_embeddings = 'Variable'
  max_length = None  # indirectly passed to config in LSTM, If defined, truncates sequences, Otherwise determined at data loading
  attention_length = 15

  ## BOW specific:
  # main params
  hidden_next = 0.6  # defines the number of hidden units in next layer
  # Determined at data loading:
  h_max_len = None  # not passed to config
  b_max_len = None  # not passed to config


def run_bow_with_parameters(args):

  # Final test st
  np.random.seed(1)
  config = Config()
  config.n_layers = 1
  config.xp = 'final_test'
  config.model = 'bow'
  config.lr = 0.005
  config.trainable_embeddings = 'Variable'
  config.b_max_len = 600
  config.n_epochs = 40
  result = run_bow(config, final = True)

  ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 1
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Variable'
  # config.b_max_len = 75
  # result = run_bow(config)

  # ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 3
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Constant'
  # config.b_max_len = 75
  # result = run_bow(config)

  # ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 0
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Variable'
  # config.b_max_len = 150
  # result = run_bow(config)

  # ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 1
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Variable'
  # config.b_max_len = 150
  # result = run_bow(config)

  # ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 3
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Variable'
  # config.b_max_len = 150
  # result = run_bow(config)

  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 0
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Variable'
  # config.b_max_len = 300
  # result = run_bow(config)

  # ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 1
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Variable'
  # config.b_max_len = 300
  # result = run_bow(config)

  # ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 3
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Constant'
  # config.b_max_len = 300
  # result = run_bow(config)

  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 0
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Variable'
  # config.b_max_len = 600
  # result = run_bow(config)

  # ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 1
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Variable'
  # config.b_max_len = 600
  # result = run_bow(config)

  # ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 3
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Constant'
  # config.b_max_len = 600
  # result = run_bow(config)

  ## Experiment 
  # np.random.seed(1)
  # config = Config()
  # config.n_layers = 3
  # config.xp = 'layers'
  # config.model = 'bow'
  # config.lr = 0.005
  # config.trainable_embeddings = 'Constant'
  # config.b_max_len = 150
  # result = run_bow(config)




def run_lstm_with_parameters(args):
  # Final test
  np.random.seed(1)
  config0 = Config()
  config0.max_length = 75
  config0.trainable_embeddings = 'Variable'
  config0.hidden_size = 100
  config0.n_epochs = 40
  config0.n_layers = 2
  config0.batch_size = 128
  config0.dropout = 0.8
  config0.lr = 0.001
  # config0.num_samples = 100
  config0.xp = 'final_test'
  config0.model = 'lstm_basic'
  result = run_lstm(config0, final = True)


  #### Testing Downsampling

  # # Experiment 1
  # # 2 layer, max_length = 75
  # np.random.seed(1)
  # config0 = Config()
  # # print('Running run_lstm_with_parameters')
  # # config0.n_layers = 0
  # config0.max_length = 75
  # config0.trainable_embeddings = 'Variable'
  # config0.hidden_size = 100
  # config0.n_epochs = 40
  # config0.n_layers = 1
  # config0.batch_size = 128
  # config0.dropout = 0.8
  # config0.n_layers = 1
  # # config0.downsample = True
  # config0.lr = 0.001
  # config0.attention_length = 15
  # result = run_lstm(config0)

  # # # # Experiment 2
  # # # # 2 layer, max_length = 150
  # np.random.seed(1)
  # config1 = Config()
  # config1.max_length = 150
  # config1.trainable_embeddings = 'Variable'
  # config1.hidden_size = 100
  # config1.n_epochs = 40
  # config1.batch_size = 128
  # config1.dropout = 0.8
  # config1.n_layers = 1
  # # config1.downsample = True
  # config1.lr = 0.001
  # config1.attention_length = 15
  # result = run_lstm(config1)

  # # ## Experiment 3
  # # # 2 layer, max_length = 300
  # np.random.seed(1)
  # config2 = Config()
  # config2.max_length = 250
  # config2.trainable_embeddings = 'Variable'
  # config2.hidden_size = 100
  # config2.n_epochs = 40
  # config2.batch_size = 128
  # config2.dropout = 0.8
  # config2.n_layers = 1
  # # config2.downsample = True
  # config2.lr = 0.001
  # config2.attention_length = 15
  # result = run_lstm(config2)

  # ## Experiment 4
  # # max_length = 150, n_layers = 1
  # np.random.seed(1)
  # config3 = Config()
  # config3.max_length = 150
  # config3.trainable_embeddings = 'Variable'
  # config3.hidden_size = 100
  # config3.n_epochs = 40
  # config3.batch_size = 128
  # config3.dropout = 0.8
  # config3.n_layers = 1
  # config3.downsample = True
  # # config3.extra_hidden_size = None
  # result = run_lstm(config3)


  # ## Experiment 5
  # # max_length = 150, n_layers = 2
  # np.random.seed(1)
  # config4 = Config()
  # config4.max_length = 150
  # config4.trainable_embeddings = 'Variable'
  # config4.hidden_size = 100
  # config4.n_epochs = 40
  # config4.batch_size = 128
  # config4.dropout = 0.8
  # config4.n_layers = 2
  # config4.downsample = True
  # result = run_lstm(config4)

  # ## Experiment 6
  # # max_length = 150, n_layers = 4
  # np.random.seed(1)
  # config5 = Config()
  # config5.max_length = 150
  # config5.trainable_embeddings = 'Variable'
  # config5.hidden_size = 100
  # config5.n_epochs = 40
  # config5.batch_size = 128
  # config5.dropout = 0.8
  # config5.n_layers = 4
  # config5.downsample = True
  # result = run_lstm(config5)


  # #### Testing Dropout

  # # ## Experiment 1
  # # # max_length = 75, n_layers = 2, dropout = 0.9
  # np.random.seed(1)
  # config = Config()
  # config.max_length = 75
  # config.trainable_embeddings = 'Variable'
  # config.hidden_size = 100
  # config.n_epochs = 40
  # config.batch_size = 128
  # config.dropout = 0.9
  # config.n_layers = 2
  # config.downsample = False
  # config.lr = 0.005
  # result = run_lstm(config)

  # # ## Experiment 2
  # # # max_length = 75, n_layers = 2, dropout = 0.65
  # np.random.seed(1)
  # config = Config()
  # config.max_length = 75
  # config.trainable_embeddings = 'Variable'
  # config.hidden_size = 100
  # config.n_epochs = 40
  # config.batch_size = 128
  # config.dropout = 0.65
  # config.n_layers = 2
  # config.downsample = False
  # config.lr = 0.005
  # result = run_lstm(config)

  # # ## Experiment 3
  # # # max_length = 75, n_layers = 2, dropout = 0.5
  # np.random.seed(1)
  # config = Config()
  # config.max_length = 75
  # config.trainable_embeddings = 'Variable'
  # config.hidden_size = 100
  # config.n_epochs = 40
  # config.batch_size = 128
  # config.dropout = 0.5
  # config.n_layers = 2
  # config.downsample = False
  # config.lr = 0.005
  # result = run_lstm(config)


  # #### Testing max_length

  # # ## Experiment 1
  # # # max_length = 50, n_layers = 2,
  # np.random.seed(1)
  # config = Config()
  # config.max_length = 50
  # config.trainable_embeddings = 'Variable'
  # config.hidden_size = 100
  # config.n_epochs = 40
  # config.batch_size = 128
  # config.dropout = 0.8
  # config.n_layers = 2
  # config.downsample = False
  # config.lr = 0.005
  # result = run_lstm(config)  

  # # ## Experiment 2
  # # # max_length = 30, n_layers = 2,
  # np.random.seed(1)
  # config = Config()
  # config.max_length = 30
  # config.trainable_embeddings = 'Variable'
  # config.hidden_size = 100
  # config.n_epochs = 40
  # config.batch_size = 128
  # config.dropout = 0.8
  # config.n_layers = 2
  # config.downsample = False
  # config.lr = 0.005
  # result = run_lstm(config)

def run_lstm_attention_with_parameters(args):
  #### Testing max_length # Experiment 1
  ## 1 layer, max_length = 50
  np.random.seed(1)
  config0 = Config()
  # print('Running run_lstm_with_parameters')
  config0.max_length = 75
  config0.trainable_embeddings = 'Variable'
  config0.hidden_size = 100
  config0.n_epochs = 40
  config0.batch_size = 128
  config0.dropout = 0.8
  config0.n_layers = 2
  config0.lr = 0.001
  config0.xp = 'final_test'
  config0.model = 'lstm_attention'
  # config0.num_samples = 100
  config0.attention_length = 15
  result = run_lstm_attention(config0, final = True)

  # np.random.seed(1)
  # config0 = Config()
  # # print('Running run_lstm_with_parameters')
  # config0.max_length = 150
  # config0.trainable_embeddings = 'Variable'
  # config0.hidden_size = 100
  # config0.n_epochs = 40
  # config0.n_layers = 2
  # config0.batch_size = 128
  # config0.dropout = 0.8
  # config0.n_layers = 4
  # # config0.downsample = False
  # config0.lr = 0.001
  # # config0.num_samples = 
  # config0.attention_length = 15
  # result = run_lstm_attention(config0)



  #### Testing attention_length # Experiment 1
  ## 1 layer, max_length = 150, attention_length = 10
  # np.random.seed(1)
  # config0 = Config()
  # # print('Running run_lstm_with_parameters')
  # config0.max_length = 150
  # config0.trainable_embeddings = 'Variable'
  # config0.hidden_size = 100
  # config0.n_epochs = 40
  # config0.n_layers = 1
  # config0.batch_size = 128
  # config0.dropout = 0.8
  # config0.n_layers = 1
  # # config0.downsample = False
  # config0.lr = 0.001
  # config0.attention_length = 10
  # result = run_lstm_attention(config0)

  # #### Testing attention_length # Experiment 2
  # ## 1 layer, max_length = 150, attention_length = 20
  # np.random.seed(1)
  # config0 = Config()
  # # print('Running run_lstm_with_parameters')
  # config0.max_length = 150
  # config0.trainable_embeddings = 'Variable'
  # config0.hidden_size = 100
  # config0.n_epochs = 40
  # config0.n_layers = 1
  # config0.batch_size = 128
  # config0.dropout = 0.8
  # config0.n_layers = 1
  # # config0.downsample = False
  # config0.lr = 0.001
  # config0.attention_length = 20
  # result = run_lstm_attention(config0)


  # #### Testing alearning rate # Experiment 1
  # ## 1 layer, max_length = 150, lr = 0.0005
  # np.random.seed(1)
  # config0 = Config()
  # # print('Running run_lstm_with_parameters')
  # config0.max_length = 150
  # config0.trainable_embeddings = 'Variable'
  # config0.hidden_size = 100
  # config0.n_epochs = 40
  # config0.n_layers = 1
  # config0.batch_size = 128
  # config0.dropout = 0.8
  # config0.n_layers = 1
  # # config0.downsample = False
  # config0.lr = 0.0005
  # config0.attention_length = 15
  # result = run_lstm_attention(config0)

  # #### Testing alearning rate # Experiment 2
  # ## 1 layer, max_length = 150, lr = 0.0002
  # np.random.seed(1)
  # config0 = Config()
  # # print('Running run_lstm_with_parameters')
  # config0.max_length = 150
  # config0.trainable_embeddings = 'Variable'
  # config0.hidden_size = 100
  # config0.n_epochs = 40
  # config0.n_layers = 1
  # config0.batch_size = 128
  # config0.dropout = 0.8
  # config0.n_layers = 1
  # # config0.downsample = False
  # config0.lr = 0.0002
  # config0.attention_length = 15
  # result = run_lstm_attention(config0)

def run_lstm_conditional_with_parameters(args):
  # To be defined - parameter saving not ready
  np.random.seed(1)
  config0 = Config()
  # print('Running run_lstm_with_parameters')
  config0.trainable_embeddings = 'Variable'
  config0.hidden_size = 100
  config0.n_epochs = 40
  config0.n_layers = 1
  config0.batch_size = 128
  config0.dropout = 0.8
  config0.n_layers = 2
  config0.lr = 0.001
  # config0.num_samples = 100
  config0.b_max_len = 75
  config0.attention_length = 15
  config0.xp = 'final_test'
  config0.model = 'conditional_lstm'
  # print 'config0' + str(config0.__dict__)
  result0 = run_lstm_conditional(config0, final = True)


  # np.random.seed(1)
  # config0 = Config()
  # # print('Running run_lstm_with_parameters')
  # # config0.n_layers = 0
  # # config0.max_length = 75
  # config0.trainable_embeddings = 'Variable'
  # config0.hidden_size = 100
  # config0.n_epochs = 40
  # config0.n_layers = 1
  # config0.batch_size = 128
  # config0.dropout = 0.8
  # config0.n_layers = 4
  # config0.lr = 0.001
  # # config0.num_samples = 100
  # config0.b_max_len = 150
  # # config0.downsample = True
  # config0.attention_length = 15
  # config0.xp = 'layers'
  # config0.model = 'conditional_lstm'
  # # print 'config0' + str(config0.__dict__)
  # result0 = run_lstm_conditional(config0)

  # np.random.seed(1)
  # config1 = Config()
  # # print('Running run_lstm_with_parameters')
  # # config0.n_layers = 0
  # # config0.max_length = 75
  # config1.trainable_embeddings = 'Variable'
  # config1.hidden_size = 100
  # config1.n_epochs = 40
  # config1.n_layers = 1
  # config1.batch_size = 128
  # config1.dropout = 0.8
  # config1.n_layers = 1
  # config1.lr = 0.001
  # # config0.num_samples = 1000
  # config1.b_max_len = 150
  # # config0.downsample = True
  # config1.attention_length = 15
  # config1.xp = 'body_length'
  # config1.model = 'conditional_lstm'
  # # print 'config0' + str(config0.__dict__)
  # result1 = run_lstm_conditional(config1)

  # np.random.seed(1)
  # config2 = Config()
  # # print('Running run_lstm_with_parameters')
  # # config0.n_layers = 0
  # # config0.max_length = 75
  # config2.trainable_embeddings = 'Variable'
  # config2.hidden_size = 100
  # config2.n_epochs = 40
  # config2.n_layers = 1
  # config2.batch_size = 128
  # config2.dropout = 0.8
  # config2.n_layers = 1
  # config2.lr = 0.001
  # # config0.num_samples = 1000
  # config2.b_max_len = 300
  # # config0.downsample = True
  # config2.attention_length = 15
  # config2.xp = 'body_length'
  # config2.model = 'conditional_lstm'
  # # print 'config0' + str(config0.__dict__)
  # result2 = run_lstm_conditional(config2)

if __name__ == "__main__":
  print("-- Running Test Script --")
  print("-- Start BOW Experiments --")	
  run_bow_with_parameters('')
  print("-- Start LSTM Basic Experiments --")	
  run_lstm_with_parameters('')
  print("-- Start LSTM Attention Experiments --")
  run_lstm_attention_with_parameters('')
  print("-- Start LSTM Conditional Experiments --")
  run_lstm_conditional_with_parameters('')
  print("-- Finished Test Script --")