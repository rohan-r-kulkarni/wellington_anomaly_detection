# deep-learning-wellington-od

## Description
\
  **Project**: Wellington Management - Multivariate Time-Series Anomaly Detection\
  **Contributors**: Lin, Ruoyu; Liu, Tianhao (Raymond); Xue, Guangrui. \
  \
  This is a project repository for 2022 Columbia Engineering IEOR4742 Deep Learning for FE and OR, Wellington Management Group 1. By the time this is written down in Dec. 2022, the group has implemented a preliminary LSTM-AutoEncoder anomaly detector. 


## Files 
- Simulation Toolkit:
  - `class_basesimulation.py`: contains base simulation class `BaseSimulation`, parent class of `Multisimulation` in `class_multisimulation`.
  - `class_multisimulation.py`: contains inherited simulation class `MultiSimulation`, child class of `BaseSimulation` in `class_basesimulation` in `class_basesimulation`. For future users: both these two classes can be modified to produce more sophisticated simulation tools.
  - `class_simulationhelper.py`: contains helper functions used to implement previous classes. It also contains other helper method for plotting.
  - `data_generation.py`: contains a single function that showcases how to automatically generate simulated datasets using only random seeds. 
  - `gen_datasets.py`: a caller file for functionalities in `data_generation.py`.
- Data:
  - `data4columbia_credit.csv`: credit card transaction data of ~50 companies, provided by Wellington Management 
  - `data4columbia_sales.csv`: quarterly sales data of ~50 companies, provided by Wellington Management
- Model:
  - `lstm_autoencoder.py`: contains two classes:
    - `DataGeneration`, a legacy class used to call simulation toolkit. Deprecated in later stage of studies.
    - `LSTM_Model_Base`, the main class to define a LSTM-AutoEncoder model. The class is designed such that an LSTM-AE architecture can be passed to initialize the model, and the class will automatically check whether the passed specification is a valid (symmetric) architecture.
  - `model_exec.py`: the main model file that contains
    - utility class `OutlierMetric` that enables the classification of outliers using different methods.
    - other helper functions that enable the execution of LSTM-AE models, see docstrings for documentation.



<!-- **Datasets:**

  gdb_by_country: comprehensive dataset for GDP by countries from 1960 - 2021 
  
  Inflation-data: inflation data by categories
  
  PCE: Personal consumer expenditure data
  
  Sale_hist: LVMH retail sales by different categories (wines ...)
  
  ...  -->
