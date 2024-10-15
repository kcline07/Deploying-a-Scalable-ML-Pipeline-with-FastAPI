# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model uses publically available census data to predict whether a person's income exceeds $50,000 by examining demographic features like education, marital status, and occupation. This model uses a logistic regression classifier method that separates individuals into two categories based on whether their income exceeds $50,000. This model showcases how to implement a machine learning pipeline using FastAPI, offering a clear, educational example of deploying models with practical applications. Standardization is used for continuous variables and one-hot encoding is used for categorical features.

## Intended Use

This model is intended to predict income levels but but has the flexibility to be adapted for other binary classification tasks.

## Training Data

The model uses a dataset of 32,561 samples from publicly available census data, with demographic features such as work class, education, marital status, and occupation. The data was split 80/20 for training and testing, and the categorical features were encoded using one-hot encoding, with labels marking whether income exceeds $50,000.

## Evaluation Data

The model was evaluated using 8,141 samples taken from the census data. The same preprocessing steps were used as in the training data and the evaluation attempted to assess the model's performance across variuous slices of demographic data.

## Metrics
The model acheivied a precison score of 0.2412, a recall score of 1.0000  and an F1 score of 0.3887. Precision measures how many of the predicted positive cases were actually correct. The model has a low precision score of 0.2412 which means that only about 24% of the predictions made by the model for income grater than $50,000 were correct. This indicates a high number of false positives. Recall measures how many of the actual positive cases the model identified successfully. A recall score of 1.0 means that this model correctly identified all individuals making over $50,000, but this comes at the cost of too many false positives. The F1 score of 0.3887 is the mean of precision and recall and provides a balanced measure of the two metrics. A value of 0.3887 suggest a low performing model, especcially due to its low precision score.

## Ethical Considerations

This model may be subject to biases that are inherent in the training data regarding demographic features such as race and gender. These biases can skew predicitons that effect certains groups more than others.

## Caveats and Recommendations

The model has high recall but low precision. The model is high senstive and can identify all high-income individuals but struggles to distinguish between those who actually earn over $50,000 and those who do not. This indicates the model is over predicting the high-income category. Methods to improve the model's precision score are recommended including tuning the decision threshold and improving feature selection and engineering. 

Addressing biases in the data is also crucial to prevent unfair predicitions for certain demographics. Bias mitigation techniques such as resampling the training data and continuously monitoring the model is recommended to alleviate ethical concerns. 