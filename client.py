import torch
import flwr as fl
from model import UNet3D, train, evaluate
from dataset_preparation import train_img_datagen, val_img_datagen, train_img_list, val_img_list


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, img_height, img_width, img_depth, img_channels, num_classes):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = UNet3D(img_height, img_width, img_depth, img_channels, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        self.model.load_state_dict({k: torch.Tensor(v) for k, v in parameters.items()}, strict=True)

    def get_parameters(self) -> dict:
        """Extract model parameters and return them as a dictionary."""
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optim, epochs, self.device)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}

def generate_client_fn(trainloader, valloader, img_height, img_width, img_depth, img_channels, num_classes):
    """Return a function that can be used by the VirtualClientEngine to spawn a FlowerClient with client id cid."""
    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloader,
            valloader=valloader,
            img_height=img_height,
            img_width=img_width,
            img_depth=img_depth,
            img_channels=img_channels,
            num_classes=num_classes,
        )
    return client_fn
