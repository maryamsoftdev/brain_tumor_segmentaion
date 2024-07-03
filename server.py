from collections import OrderedDict
from omegaconf import DictConfig
import torch
from model import UNet3D, train, evaluate


def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # configure_fit() method.

        # Here we are returning the same config on each round but
        # here you might use the server_round input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(img_height, img_width, img_depth, img_channels, num_classes, valloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        # This function is called by the strategy's evaluate() method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on an evaluation / validation dataset.

        model = UNet3D(img_height, img_width, img_depth, img_channels, num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Evaluate the global model on the validation set
        loss, accuracy = evaluate(model, valloader, device)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global validation accuracy.
        return loss, {"accuracy": accuracy}

    return evaluate_fn
