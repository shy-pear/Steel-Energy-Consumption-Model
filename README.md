# Steel Energy Consumption Model
## Project Summary
This project tests versions of the stacked LSTM model to determine proficiency in predicting future energy consumption trends by adjusting model framework and hyperparameters. A stacked LSTM model with three LSTM layers and one dense layer is used as a baseline. Model is compiled with standard gradient descent optimizer, and loss function is mean squared error.
### Tests:
Altering optimizer and monitoring learning rate: Optimizer changed to Adam (Adaptive Moment Estimation) to allow for more dynamic learning rate optimization. Default learning rate is reduced from default 0.001 to 0.0005, as time-series data may be particularly sensitive to abrupt changes in weight. Learning rate is monitored during training process, and adjusted according to val loss and training loss. As val loss flattens and diverges from training loss, epochs are stopped and learning rate is reduced to 0.0001. Training resumes until epochs finish or val loss and training loss show no signs of improvement, at which point training is manually stopped and tested.

Adjusting dropout layers: A dropout layer with a rate of 0.2 is added after each LSTM layer to help prevent overfitting and improve model robustness. Training is set to 50 epochs and allowed to run.
### Results:
The original LSTM model performed poorly, with val loss not decreasing exponentially as expected. Test data RMSE was 0.19, and the R squared score was -0.018. This showed that the model was not able to capture the relationship between features and create a meaningful model.

![img1](https://github.com/user-attachments/assets/c36adaf3-e675-4e95-abb5-80b05c3396ba)

For the second test, the optimizer was changed and the learning rate was monitored during training through tensorboard. 

![img2](https://github.com/user-attachments/assets/75af7385-51f9-4f74-9811-9fa3d1450d1b)

This figure shows a much better trend for both training and validation loss. Both decreased quickly at the beginning as expected and trended towards convergence. The val loss remained volatile at times. The test data RMSE score was 0.069 and the R squared value was 0.87.

![img3](https://github.com/user-attachments/assets/89f00f1f-e822-43cd-8daa-7a457f842839)

When the model was tested by predicting the energy consumption during the last 100 days, however, it produced predictions similar to the original values but not exact.

In the last test, the optimizer was set to Adam and the learning rate to 0.0001 as this proved to be most effective during the previous test. Dropout layers were also added in between LSTM layers.

![img4](https://github.com/user-attachments/assets/22b4da80-682d-4f22-8b73-a11f43e4c3c5)

This shows a generally solid trend of val loss and training loss. It is slower than the previous model, and the two functions do not cross paths. However they do trend downwards and generally towards convergence. The test data RMSE score was 0.086 and R squared value was 0.81.

![img5](https://github.com/user-attachments/assets/0c27488c-87bc-4716-a4c7-7fe05d6d106b)

This test shows the most accurate matchup of predictions and actual data with a few major skews.

Overall, the results show that for this set of data, a lower learning rate with the Adam optimizer results in meaningful training of the model, and that an increase in dropout layers generates better predictions. However, the number of dropout layers should be further investigated to increase the RMSE and R Squared scores, which decreased from the previous trial.

## Motivation and Goal
Predicting energy efficiency is important for companies as it allows them to anticipate and control energy expenses. Specifically for industries that heavily rely on industrial processes such as steel production, energy consumption can become a major expense as well as a potential hazard to the environment. Accurate forecasts can help companies identify potential shortcomings in their production processes and plan to meet regulations and expense goals.

LSTMS are specifically designed to handle sequential dependencies, unlike traditional RNNs. Their ability to selectively retain relevant information long-term, as well as avoid vanishing gradients, make them a robust choice for this temporal energy consumption data. The goal is to test and adjust hyperparameters while closely monitoring the model to produce the most accurate predictions.

## Tools and Datasets
### Tools:
- Jupyter notebook
- Numpy
- Sci-kit learn
- Tensorflow Keras
- Tensorboard
- Matplotlib
### Dataset:
Steel Industry Energy Consumption data
- Data gathered from DAEWOO Steel Co, a South Korean company that produces coils, steel plates, and iron plates.
- Information on daily energy consumption of the company during the year 2018,
including reactive power and current power factors

## Limitations and Next Steps
The R squared value is currently around 0.8 and predictions are not completely accurate, and there are signs that the data may be underfitting. There might be several issues, such the model is currently too simple (not enough layers or neurons), not trained long enough (epochs), or may be using the wrong time step or batch size. Tests can be conducted to adjust these hyperparameters. In terms of learning rate, I want to further explore how learning rate schedulers to automatically change learning rate without manual interference.

## Credits
V E, S., Shin, C., & Cho, Y. (2021). Steel Industry Energy Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C.

