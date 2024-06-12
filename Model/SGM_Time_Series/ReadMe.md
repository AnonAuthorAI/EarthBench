## ReadMe

- Run __data_split.py__ first to process the raw time series data from ../../Data/Labeled_Time_Series. The processed data will be stored in the default folder processed_weather_data/
- Then, run __re_feature.py__ to do the feature engineering to pave the way for finding location-wise causality based on (1) time and (2) feature generation
- After that, run __finding_causality.py__
- Then, run __forecasting.py__ to train the forecasting module on the training set
- Finally, run __evaluation.py__ to test the forecasting performance and anomaly detection in the forecast

### Requirement
- pandas == 1.4.2
- torch == 1.12.1


### Acknowledgment
- [Multivariate Time-series Anomaly Detection via Graph Attention Network](https://github.com/mangushev/mtad-gat)
- [DAG-GNN: DAG Structure Learning with Graph Neural Networks](https://github.com/fishmoon1234/DAG-GNN)
- [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://github.com/liyaguang/DCRNN)
