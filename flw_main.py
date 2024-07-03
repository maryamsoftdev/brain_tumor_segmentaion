import pickle
from pathlib import Path
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
import torch
from client import generate_client_fn
from custom_datagenerator import imageLoader
from model import UNet3D, train, evaluate
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    # Load dataset
    base_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128"
    train_img_dir = os.path.join(base_dir, "train", "images")
    train_mask_dir = os.path.join(base_dir, "train", "masks")
    val_img_dir = os.path.join(base_dir, "val", "images")
    val_mask_dir = os.path.join(base_dir, "val", "masks")

    train_img_list = os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)
    val_img_list = os.listdir(val_img_dir)
    val_mask_list = os.listdir(val_mask_dir)

    batch_size = cfg.batch_size
    trainloader = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
    valloader = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

    print(len(train_img_list), len(train_mask_list))
    img_height = cfg.img_height
    img_width = cfg.img_width
    img_depth = cfg.img_depth
    img_channels = cfg.img_channels
    # Generate client function
    client_fn = generate_client_fn(trainloader=trainloader, valloader=valloader,
                                   num_classes=cfg.num_classes,
                                   img_height=img_height,
                                   img_width=img_width,
                                   img_depth=img_depth,
                                   img_channels=img_channels)
    # Instantiate a client with client ID "0"
    client = client_fn(cid="0")

    # Define federated learning strategy
    strategy = fl.server.strategy.FedProx(
        fraction_fit=0.0,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.0,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.img_height, cfg.img_width, cfg.img_depth, cfg.img_channels, cfg.num_classes, valloader),
        proximal_mu=0.1  # Add the proximal term parameter
    )

    # Start federated learning simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },
    )

    # Save results
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # Execute main function
    main()
