# Bayesian-Optimization-for-DNN-hyperparameter-tuning-

Hyperparameter tuning can be a time-consuming task, particularly when using traditional methods such as GridSearch and Random Search, which explore the entire range of possible parameter values independently without considering previous results. This becomes increasingly challenging as the number of parameters to be tuned increases, as the search space grows exponentially and requires training a model, generating predictions, and calculating the validation metric for each combination of hyperparameters.

Bayesian Optimization, also known as Bayesian reasoning, can greatly reduce the time and effort required for hyperparameter tuning while also improving the generalization performance on the test set. This method takes into account the information from previously evaluated hyperparameter combinations when deciding on the next set to evaluate, making the search process more efficient and effective.

## Hyperparameter fora deep nural network trained from End-to-End
- This method uses CNN-LSTM-CTC networks.
- The metrics is validation loss.
- To select the best hyperparameters, train the model, and then test its performace; please run the below codes in your terminal

### To run the code with Terminal use the following info:
```
# Load and Pre-process data
python3 data_loader.py

# Train
##to select the best model  hyperparameters
1. python3 train_model_BBO_pre_select.py 
##to train the model with the baysian suggested hyperparameters
2. python3 train_model_BBO_best.py

# Test and results
python3 test_model_BBO.py
```
## Some issues to know
1. The test environment is
    - Python 3.8+
    - Keras 2.2.4
    - tensorflow 2+
