import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ml.data import process_data

# TODO: add necessary imports (if required)

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    scaler = StandardScaler()  # Fixed variable name
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=5000, solver='liblinear')  # Fixed model name
    model.fit(X_train_scaled, y_train)
    
    return model

"""
Trains a machine learning model and returns it.

Inputs
------
X_train : np.array
    Training data.
y_train : np.array
    Labels.
Returns
-------
model
    Trained machine learning model.
"""


def compute_model_metrics(y, preds):
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    
    return precision, recall, fbeta

"""
Validates the trained machine learning model using precision, recall, and F1.

Inputs
------
y : np.array
    Known labels, binarized.
preds : np.array
    Predicted labels, binarized.
Returns
-------
precision : float
recall : float
fbeta : float
"""


def inference(model, X):
    preds = model.predict(X)
    return preds

"""
Run model inferences and return the predictions.

Inputs
------
model : ???
    Trained machine learning model.
X : np.array
    Data used for prediction.
Returns
-------
preds : np.array
    Predictions from the model.
"""


def save_model(model, path):
    with open(path, 'wb') as f:  # Fixed variable name to 'path'
        pickle.dump(model, f)

"""
Serializes model to a file.

Inputs
------
model
    Trained machine learning model or OneHotEncoder.
path : str
    Path to save pickle file.
"""


def load_model(path):
    with open(path, 'rb') as f:  # Fixed variable name to 'path'
        model = pickle.load(f)
    return model

"""
Loads pickle file from `path` and returns it.
"""


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes the model metrics on a slice of the data specified by a column name and slice value.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features.
    label : str
        Name of the label column in `X`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    
    # Filter the data based on the column and slice value
    data_slice = data[data[column_name] == slice_value]

    # Process the data slice
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Get predictions for the slice
    preds = inference(model, X_slice)
    
    # Compute the metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    
    return precision, recall, fbeta

"""
Computes performance metrics (precision, recall, fbeta) on a specified slice of data.

Inputs
------
data : pd.DataFrame
    The dataset.
column_name : str
    The name of the column to slice.
slice_value : str/int/float
    The value of the feature for slicing.
categorical_features: list
    Categorical features in the dataset.
label : str
    Target label name.
encoder : OneHotEncoder
    Encoder used to encode categorical features.
lb : LabelBinarizer
    Encoder used to encode the target label.
model : ???
    Trained model to generate predictions.
Returns
-------
precision : float
recall : float
fbeta : float
"""

