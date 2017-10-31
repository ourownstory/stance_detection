CS 224n Project Directory

Winter 2017

Stephen Pfohl
Ferdinand Legros
Oskar Triebe

Model Files:
    our_model_config.py
        contains abstract model class to be extended by other models. Is based off of the model classes used in the course assignments.
    bow_model_config.py
        Bag of words model class that extends our_model_config.py
    basicLSTM_model_config.py
        model class for the basic LSTM model that operates on the concatenated input
    LSTM_attention.py
        model class for the LSTM model that has been augmented by attention
    LSTM_conditional.py
        model class for the LSTM with attention and conditional encoding

Model Execution Files
    execute_bow_config
        script that executes a single experiment of the bag of words model for a given set of parameters
    execute_lstm_config.py
        script that executes a single experiment of the basic LSTM model for a given set of parameters
    execute_lstm_attention.py
        script that executes a single experiment of the lstm model that has been augmented by attention for a given set of parameters
    execute_lstm_conditional.py
        script that executes a single experiment of the LSTM model with conditional encoding and attention for a given set of parameters for a given set of parameters

Utility Files
    our_util.py
        Utility functions for use in other files. Based on the example of the util.py files provided in course assignments.
    run_text_processing.py
        File that performas tokenization, loads the data, etc

Runtime scripts
    test_script6.py
        Allows the user to define a set of experiments for any of the models described above.

fnc_baseline directory
    Required and provided by the competition organizers at https://github.com/FakeNewsChallenge/fnc-1-baseline
    Not included with this submission due to size constraints

Plotting
    Contains .Rmd files for plotting