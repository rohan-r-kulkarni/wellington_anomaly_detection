# deep-learning-wellington-od

## Description
\
  **Project**: Wellington Management - Multivariate Time-Series Anomaly Detection\
  **Contributors**: Rohan R. Kulkarni, Ethan Li, Ruoyu Lin, Tianhao (Raymond) Liu, Guangrui Xue\
  **Supervisors**: Daniel Fernandez, Yossi Cohen (Wellington Management), Ali Hirsa (Columbia IEOR)
  \
  This is a project repository for 2022-2023 Columbia Engineering AI Applications in Finance Wellington Management (group 1).

## Files 
- Data (`./data`):
  - `data4columbia_credit.csv`: credit card transaction data of ~50 companies, provided by Wellington Management 
  - `data4columbia_sales.csv`: quarterly sales data of ~50 companies, provided by Wellington Management
  - `featured_credit.csv`: credit data set after processing and feature engineering. The actual data set fed to the model.
- EDA (`./eda`):
  - `class_singlecompanyeda.py`: contains EDA class SingleCompanyEDA for the EDA pipeline of a single company.
  - `class_multicompanyeda.py`: contains EDA class MultiCompaniesEDA for the EDA pipeline of multiple companies.
  - `wellington_data_processing.py`: the data processing pipeline for the credit card data set.
  - `EDA_demo.ipynb`: the EDA pipeline aggregated.
  - `EDA_credit.ipynb`: EDA conducted on the credit dataset.
  - `EDA_sales.ipynb`: EDA conducted on the sales dataset.
  - `bayesian.ipynb`: EDA for the Bayesian hierarchical model.
- Model (`./model`):
  - `__init__.py`: for modularization purpose only.
  - `lstm_autoencoder.py`: contains two classes:
    - `DataGeneration`, a legacy class used to call simulation toolkit. Deprecated in later stage of studies, but still salvagable for future simulations.
    - `LSTM_Model_Base`, the main class to define a LSTM-AutoEncoder model. The class is designed such that an LSTM-AE architecture can be passed to initialize the model, and the class will automatically check whether the passed specification is a valid (symmetric) architecture.
  - `lstm_windows.py`: contains two classes:
    - `LSTMWindows`, the main class for LSTM windows analysis that defines all the training and testing methods for running brute force or STL-assisted LSTM anomaly detection in rolling windows.
    - `LSTMWindowPlot`, the plotting class that contains helper methods to easily plot results of deep learning methods, original time series and reconstructions of the LSTM autoencoder, and more.
  - `model_exec.py`: the main model execution file that contains
    - utility class `OutlierMetric` that enables the classification of outliers using different methods.
    - other helper functions that enable the execution of LSTM-AE models, see docstrings for documentation.
- `run_windows.py`: runs the LSTM rolling windows brute force method for anomaly detection.
- `run_stl_windows.py`: runs the STL-assisted LSTM rolling windows method for anomaly detection.
- `plot_windows.py`: plots and/or saves results and analysis from a LSTM windows train/test session.
- Bayesian and Wavelet models (`./bayes_wavelet`):
  - `readme.md`: readme for using these models.
  - `requirements.txt`: Python environment requirements for using these models.
  - `./data`: data for using these models in local reference.
  - `bayesian.py`: defines the architecture of the Bayesian Hierarchical model.
  - `bayesian_adjusted.py`: implementation of the Bayesian model.
  - `bayesian_partial_pool.py`: defines the architecture of the Bayesian partially-pooled model.
  - `bayesian_unpooled.py`: defines the architecture of the Bayesian unpooled model.
  - `wavelet.ipynb`: implementation of the wavelet model and analysis.
- LSTM Windows Results (`./lstm_windows_res`):
    - Contains text files of reconstructions, anomalous designations, and other data from LSTM windows train/test sessions.
    - `./plots`: generated plots from LSTM windows brute force and STL-assisted analysis.
    - `./anom_plots`: generated plots of anomalous regions and reconstructions from LSTM windows brute force and STL-assisted analysis.
- Simulation Toolkit (`./sim_util`):
  - `__init__.py`: for modularization purpose only.
  - `class_basesimulation.py`: contains base simulation class `BaseSimulation`, parent class of `Multisimulation` in `class_multisimulation`.
  - `class_multisimulation.py`: contains inherited simulation class `MultiSimulation`, child class of `BaseSimulation` in `class_basesimulation` in `class_basesimulation`. For future users: both these two classes can be modified to produce more sophisticated simulation tools.
  - `class_simulationhelper.py`: contains helper functions used to implement previous classes. It also contains other helper method for plotting.
  - `data_generation.py`: contains a single function that showcases how to automatically generate simulated datasets using only random seeds. 
  - `gen_datasets.py`: a caller file for functionalities in `data_generation.py`.
- LSTM Model Artifacts (`./lstm_model`):
    - Contains Tensorflow/keras model saves, sessions, and artifacts for the deep learning models used.
- `demo_2022.ipynb`: the 2022 demo notebook, kept for completeness.
- Deprecated (`./deprecated`):
    - Contains deprecated but perhaps still useful code.

