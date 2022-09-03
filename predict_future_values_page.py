from urllib.error import URLError
import streamlit as st
import pandas as pd

import datetime
import time
import pandas as pd 


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


import plotly.graph_objs as go
from plotly.offline import iplot

from datetime import date
import holidays

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from torch.utils.data import TensorDataset, DataLoader

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.optim as optim


def plot_dataset(df, title):
    data = []
    
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(180,180,0, 1)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    #iplot(fig)
    st.plotly_chart(fig, use_container_width=True)

def onehot_encode_pd(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col  )
        
    return pd.concat([df, dummies], axis=1)
    #return pd.concat([df, dummies], axis=1).drop(columns=cols)

def generate_other_features(df, col_name , n, m):
    for i in range(n,m+1):
        if f'{col_name}_{i}'  not in df.columns:

                kwargs = {
                f'{col_name}_{i}' : lambda x: 0
                    }    
                df = df.assign(**kwargs) 
    return df
def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
             }
    return df.assign(**kwargs)

us_holidays = holidays.US()
def is_holiday(date):
                date = date.replace(hour = 0)
                return 1 if (date in us_holidays) else 0
def add_holiday_col(df, holidays):
                return df.assign(is_holiday = df.index.to_series().apply(is_holiday))

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y
def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_scaler(scaler):
        scalers = {
                "minmax": MinMaxScaler,
                "standard": StandardScaler,
                "maxabs": MaxAbsScaler,
                "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)

def plot_predictions(df_result  ,df_result1, df_result2, df_baseline):
    data = []
    
    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    baseline = go.Scatter(
        x=df_baseline.index,
        y=df_baseline.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='linear regression',
        marker=dict(),
        text=df_baseline.index,
        opacity=0.8,
    )
    data.append(baseline)
    
    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='predictions lstm',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)

    prediction1 = go.Scatter(
        x=df_result1.index,
        y=df_result1.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='predictions rnn',
        marker=dict(),
        text=df_result1.index,
        opacity=0.8,
    )
    data.append(prediction1)

    prediction2 = go.Scatter(
        x=df_result2.index,
        y=df_result2.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='predictions gru',
        marker=dict(),
        text=df_result2.index,
        opacity=0.8,
    )
    data.append(prediction2)
    
    layout = dict(
        title="Predictions vs Actual Values for the dataset",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)
    
def plot_predictions1(df_result  , df_baseline , model_name):
    data = []
    
    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(180,100,0, 1)"),
    )
    data.append(value)

    baseline = go.Scatter(
        x=df_baseline.index,
        y=df_baseline.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='linear regression',
        marker=dict(),
        text=df_baseline.index,
        opacity=0.8,
    )
    data.append(baseline)
    
    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='predictions '+model_name,
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)

    
    
    layout = dict(
        title="Predictions "+model_name  +" vs Actual Values for the dataset",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    #iplot(fig)
    st.plotly_chart(fig, use_container_width=True)

    


def build_baseline_model(df, test_ratio, target_col):
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result

def calculate_metrics(df):
    result_metrics = {'mae' : mean_absolute_error(df.value, df.prediction),
                      'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
                      'r2' : r2_score(df.value, df.prediction)}
    
    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df

def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result

def plot_dataset_with_forecast(df, df_forecast, df_forecast1, df_forecast2, title):
    data = []
    
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    forecast = go.Scatter(
        x=df_forecast.index,
        y=df_forecast.prediction,
        mode="lines",
        name="forecasted values LSTM",
        marker=dict(),
        text=df.index,
        opacity=0.8,

        #line=dict(color="rgba(10,100,10, 0.3)"),
    )
    data.append(forecast)

    forecast1 = go.Scatter(
        x=df_forecast1.index,
        y=df_forecast1.prediction,
        mode="lines",
        name="forecasted values RNN",
        marker=dict(),
        text=df.index,
        opacity=0.8,

    )
    data.append(forecast1)

    forecast2 = go.Scatter(
        x=df_forecast2.index,
        y=df_forecast2.prediction,
        mode="lines",
        name="forecasted values GRU",
        marker=dict(),
        text=df.index,
        opacity=0.8,

    )
    data.append(forecast2)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)
def plot_dataset_with_forecast1(df, df_forecast, title , model_name):
    data = []
    
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(100,180,0, 1)"),
    )
    data.append(value)

    forecast = go.Scatter(
        x=df_forecast.index,
        y=df_forecast.prediction,
        mode="lines",
        name="forecasted values "+model_name,
        marker=dict(),
        text=df.index,
        opacity=0.8,

        #line=dict(color="rgba(10,100,10, 0.3)"),
    )
    data.append(forecast)

    

    layout = dict(
        title=title +' '+  model_name,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    #iplot(fig)
    st.plotly_chart(fig, use_container_width=True)


def format_forecasts(forecasts, index, scaler):
    preds = np.concatenate(forecasts, axis=0).ravel()
    df_forecast = pd.DataFrame(data={"prediction": preds}, index=index)
    df_result = df_forecast.sort_index()
    df_result = inverse_transform(scaler, df_result, [["prediction"]])
    return df_result

def get_datetime_index(df):
    return (
        pd.to_datetime(df.index[-1])
        + (pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[-2])),
        pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[-2]),
    )


### Vanilla RNN
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates an RNN instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out
### Long Short-Term Memory (LSTM) 
class LSTMModel(nn.Module):
    """LSTMModel class extends nn.Module class and works as a constructor for LSTMs.

       LSTMModel class initiates a LSTM module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           lstm (nn.LSTM): The LSTM model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of LSTMs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a LSTM instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
### Gated Recurrent Unit (GRU)
class GRUModel(nn.Module):
    """GRUModel class extends nn.Module class and works as a constructor for GRUs.

       GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           gru (nn.GRU): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
#Helper/Wrapper Class for training
class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.

    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.

    Attributes:
        model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """
    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        model_path = f'{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        #torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def forecast_with_predictors(
        self, forecast_loader, batch_size=1, n_features=1, n_steps=100
    ):
        """Forecasts values for RNNs with predictors and one-dimensional output

        The method takes DataLoader for the test dataset, batch size for mini-batch testing,
        number of features and number of steps to predict as inputs. Then it generates the
        future values for RNNs with one-dimensional output for the given n_steps. It uses the
        values from the predictors columns (features) to forecast the future values.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns
            n_steps (int): Number of steps to predict future values

        Returns:
            list[float]: The values predicted by the model

        """
        step = 0
        with torch.no_grad():
            predictions = []
            for x_test, _ in forecast_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())

                step += 1
                if step == n_steps:
                    break

        return predictions

    def forecast_with_lag_features(self, test_loader, batch_size=1, n_features=1, n_steps=100):
        """The method forecast() forecasts values for RNNs with one-dimensional output

        The method takes DataLoader for the test dataset, batch size for mini-batch testing,
        number of features and number of steps to predict as inputs. Then it generates the
        future values for RNNs with one-dimensional output for the given n_steps.

        It uses the last item from the Test DataLoader to create the next input tensor (X).
        First, it shifts the tensor by one and adds the actual value from the target tensor (y).
        For the given n_steps, it goes on to predict the values for the next step and to update
        the input tensor for the next prediction. During each iteration it stores the predicted
        values in the list.

        Note:
            Unlike the method evaluate: This method does not assume that the prediction from
            the previous step is available the time of the prediction. Hence, it takes the
            first item in the Test DataLoader then only does one-step prediction into the future.
            After predicting the value for the next step, it updates the input tensor (X)
            by shifting the tensor and then adding the most recent prediction.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns
            n_steps (int): Number of steps to predict future values


        Returns:
            list[float]: The values predicted by the model

        """
        test_loader_iter = iter(test_loader)
        predictions = []

        *_, (X, y) = test_loader_iter

        y = y.to(device).detach().numpy()
        X = X.view([batch_size, -1, n_features]).to(device)
        X = torch.roll(X, shifts=1, dims=2)
        X[..., -1, 0] = y.item(0)

        with torch.no_grad():
            self.model.eval()
            for _ in range(n_steps):
                X = X.view([batch_size, -1, n_features]).to(device)
                yhat = self.model(X)
                yhat = yhat.to(device).detach().numpy()
                X = torch.roll(X, shifts=1, dims=2)
                X[..., -1, 0] = yhat.item(0)
                predictions.append(yhat.item(0))

        return predictions
    
    def plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

device = "cuda" if torch.cuda.is_available() else "cpu"

def features_(df1):
        ## Generating date/time predictors
        df_features = (
                df1
                .assign(hour = df1.index.hour)
                .assign(day = df1.index.day)
                .assign(month = df1.index.month)
                .assign(day_of_week = df1.index.dayofweek)
                .assign(week_of_year = df1.index.week)
              )
        ### One-hot encoding
        df_features = onehot_encode_pd(df_features, ['day'])
        df_features = generate_other_features(df_features, 'day' , 1 ,31)
        df_features = onehot_encode_pd(df_features, ['month'])
        df_features = generate_other_features(df_features, 'month' ,1 ,  12)
        df_features = onehot_encode_pd(df_features, ['day_of_week'])
        df_features = generate_other_features(df_features, 'day_of_week'  , 0, 6)
        df_features = onehot_encode_pd(df_features, ['week_of_year'])
        df_features = generate_other_features(df_features, 'week_of_year' , 0 , 56)
        ### Generating cyclical features (sin/cos transformation)
        df_features = generate_cyclical_features(df_features, 'hour', 24, 0)
        df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
        df_features = generate_cyclical_features(df_features, 'month', 12, 1)
        df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)
        ## Other date/time-related features        
        df_features = add_holiday_col(df_features, us_holidays)
        return df_features

def forecasts_(df_forecast):
    df_forecast['value'] = 0

    df_forecast= (df_forecast
                .assign(hour = df_forecast.index.hour)
                .assign(day = df_forecast.index.day)
                .assign(month = df_forecast.index.month)
                .assign(day_of_week = df_forecast.index.dayofweek)
                .assign(week_of_year = df_forecast.index.week)
                )

    df_forecast = onehot_encode_pd(df_forecast, ['month','day','day_of_week','week_of_year'])
    df_forecast = generate_cyclical_features(df_forecast, 'hour', 24, 0)

    df_forecast = generate_cyclical_features(df_forecast, 'day_of_week', 7, 0)
    df_forecast = generate_cyclical_features(df_forecast, 'month', 12, 1)
    df_forecast = generate_cyclical_features(df_forecast, 'week_of_year', 52, 0)

    df_forecast = add_holiday_col(df_forecast, us_holidays)


    return df_forecast



def indexof(names , name):
    for i in range(len(names)):
     if names[i] == name:
        return i 


def show_predict_future_values_page(save_data , save_name):
    dataframe_collection = save_data
    names_tables = save_name
    st.title("Predict future values")
    df = pd.read_csv('FACEBOOK.csv')
    i=0
    dataframe_collection[i]= df
    names_tables[i] = 'FACEBOOK'

    df = pd.read_csv('INSTAGRAM.csv')
    i=i+1
    dataframe_collection[i]= df
    names_tables[i] = 'INSTAGRAM'

    df = pd.read_csv('LINKEDIN.csv')
    i=i+1
    dataframe_collection[i]= df
    names_tables[i] = 'LINKEDIN'

    


    if(len(dataframe_collection) != 0 and len(names_tables) !=0):
       
        
        try:


            names = st.multiselect(
                "Choose tables", list(names_tables.values()), names_tables[0]
            )

            for name in names : 
                i = indexof(names_tables , name)                    
                df1 = save_data[i]
                df =pd.DataFrame(data=df1) 
                df = df.set_index(['created_timestamp'])
                df.index = pd.to_datetime(df.index)
                if not df.index.is_monotonic:
                    df = df.sort_index()

                #st.write("names_tables : ",names_tables[i])
                st.write("name : ",name)
                features = st.multiselect(
                    "Choose features", list(df.columns), list(df.columns)[len(df.columns)-1]
                )
                if not features:
                    st.error("Please select at least one feature.")
                else:
                    
                    for feature in features : 
                        df1 = df.rename(columns={feature: 'value'})
                        plot_dataset(df1, title=name +'-'+ feature)
        
                        df_features = features_(df1)
                        ## Splitting the data into test, validation, and train sets
                        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.2)
                        #Applying scale  transformation
                        scaler = get_scaler('minmax')
                        X_train_arr = scaler.fit_transform(X_train)
                        X_val_arr = scaler.transform(X_val)
                        X_test_arr = scaler.transform(X_test)

                        y_train_arr = scaler.fit_transform(y_train)
                        y_val_arr = scaler.transform(y_val)
                        y_test_arr = scaler.transform(y_test)

                        ### Loading the data into DataLoaders
                        batch_size = 90

                        train_features = torch.Tensor(X_train_arr)
                        train_targets = torch.Tensor(y_train_arr)
                        val_features = torch.Tensor(X_val_arr)
                        val_targets = torch.Tensor(y_val_arr)
                        test_features = torch.Tensor(X_test_arr)
                        test_targets = torch.Tensor(y_test_arr)

                        train = TensorDataset(train_features, train_targets)
                        val = TensorDataset(val_features, val_targets)
                        test = TensorDataset(test_features, test_targets)

                        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
                        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
                        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
                        test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
                        
                        ### Training the model
                        input_dim = len(X_train.columns)
                        output_dim = 1
                        hidden_dim = 1
                        layer_dim = 1#3
                        batch_size = 90
                        dropout = 0.2
                        n_epochs = 350
                        learning_rate = 1e-3
                        weight_decay = 1e-6
                        model_params = {'input_dim': input_dim,
                        'hidden_dim' : hidden_dim,
                        'layer_dim' : layer_dim,
                        'output_dim' : output_dim,
                        'dropout_prob' : dropout}

                        models_names = st.multiselect(
                        "Choose models", ['lstm' , 'rnn' , 'gru' ], 'lstm',key=feature
                         )

                        for model_name in models_names : 
                            model = get_model(model_name, model_params)
                            loss_fn = nn.MSELoss(reduction="mean")
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                            opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
                            opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
                            opt.plot_losses()
                            predictions, values = opt.evaluate(
                                test_loader_one,
                                batch_size=1,
                                n_features=input_dim
                            )
                            df_result = format_predictions(predictions, values, X_test, scaler)
                            result_metrics = calculate_metrics(df_result)
                            #st.write('error')
                            #st.write(result_metrics)
                            ##
                            err = mean_absolute_error(df_result.value, df_result.prediction)
                            df_result.prediction = df_result.prediction + err

                            df_baseline = build_baseline_model(df_features, 0.2, 'value')
                            baseline_metrics = calculate_metrics(df_baseline)
                            ##
                            err1 = mean_absolute_error(df_baseline.value, df_baseline.prediction)
                            df_baseline.prediction = df_baseline.prediction + err1
                            ## 
                            plot_predictions1(df_result, df_baseline , model_name)


                            #### Predicting future values
                            NS = 365
                            start_date, freq = get_datetime_index(y_test)
                            index = pd.date_range(start=start_date, freq=freq, periods=NS )
                            df_forecast = pd.DataFrame(index=index)
                            df_forecast = forecasts_(df_forecast)
                            df_forecast[list(set(X_train.columns)-set(df_forecast.columns))] = 0
                            
                            X_forecast, y_forecast = feature_label_split(df_forecast, 'value')
                            scaler = get_scaler('minmax')
                            X_train_arr = scaler.fit_transform(X_train)
                            X_forecast_arr = scaler.transform(X_forecast)
                            y_train_arr = scaler.fit_transform(y_train)
                            y_forecast_arr = scaler.transform(y_forecast)

                            forecast_dataset = TensorDataset(torch.Tensor(X_forecast_arr), 
                                                            torch.Tensor(y_forecast_arr))
                            forecast_loader = DataLoader(
                                forecast_dataset, batch_size=1, shuffle=False, drop_last=True
                            ) 
                            forecasts = opt.forecast_with_predictors(forecast_loader, 
                                            batch_size=1, 
                                            n_features=input_dim,
                                            n_steps=NS)
                            df_forecast = format_forecasts(forecasts, index, scaler)
                            df_forecast.prediction = df_forecast.prediction + err
                            title='Predict future values'
                            plot_dataset_with_forecast1(df1, df_forecast , title , model_name)



        
        except URLError as e:
            st.error(
                """
                **This app requires internet access.**
                Connection error: %s
            """
                % e.reason
            )



