from typing import Dict, Optional, Tuple
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import flwr as fl
import tensorflow as tf
import matplotlib.pyplot as plt


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    model = Sequential()
    model.add(Dense(11,activation='relu',input_dim=13))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy
    )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train), _ = load_partition()

    # Use the last 50 training examples as a validation set
    x_val, y_val = x_train[:50], y_train[:50]

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)

        global loss_history
        loss_history.append(loss)
        global accuracy_history
        accuracy_history.append(accuracy)
        global round_history
        round_history.append(server_round)

        return loss, {"accuracy": accuracy}

    return evaluate


def load_partition():
    dataset = pd.read_csv("heart.csv")

    predictors = dataset.drop("target",axis=1)
    target = dataset["target"]

    X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=42)
    return (X_train, Y_train), (X_test, Y_test)


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 100 if server_round < 2 else 300,
        # "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def plot1(round_history,loss_history):
    
    plt.plot(round_history,loss_history)
    plt.title('rounds vs loss')
    plt.xlabel('round')
    plt.ylabel('loss')
    plt.show()
    
def plot2(round_history,accuracy_history):
    plt.plot(round_history,accuracy_history)
    plt.title('rounds vs accuracy')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.show()

loss_history=[]
accuracy_history=[]
round_history=[]



if __name__ == "__main__":
    main()
    
plot1(round_history,loss_history)
plot2(round_history,accuracy_history)