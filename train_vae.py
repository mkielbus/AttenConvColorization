from pyaiwrap.train import train
from pyaiwrap.config import buildNeuralNetworkFromJson
from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.loss import VAELoss
from pyaiwrap.metrics import VAEMetrics
from pyaiwrap.vae import loadHyperparameters
from pyaiwrap.control import vaeControlFunction
from pyaiwrap.transforms import ToGrayscale, ExtractGreenChannelTo3Channel
from pyaiwrap.utils import prepareDevice
from pyaiwrap.neural_network import VAE
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse


def parseCMDArgs():
    parser = argparse.ArgumentParser(description="Train VAE with configurable hyperparameters.")
    parser.add_argument(
        "--hyperparams",
        type=str,
        required=False,
        default="./hyperparams_vae/0.json",
        help="Path to the JSON file containing hyperparameters"
    )
    parser.add_argument(
        "--launch_number",
        type=int,
        required=False,
        default=0,
        help="The number of the training process launch with the same hyperparams file (increase it for subsequent runs\
with the same hyperparams file)."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    device = prepareDevice()
    args = parseCMDArgs()

    hyperparams = loadHyperparameters(args.hyperparams)
    BATCH_SIZE = hyperparams["BATCH_SIZE"]
    TRAIN_DATA_PATH = hyperparams["TRAIN_DATA_PATH"]
    VALIDATION_DATA_PATH = hyperparams["VALIDATION_DATA_PATH"]
    IMAGE_RESIZE = hyperparams["IMAGE_RESIZE"]
    INPUT_CHANNELS = hyperparams["INPUT_CHANNELS"]
    HYPERPARAMS_ID = hyperparams["HYPERPARAMS_ID"]
    ARCHITECTURE_ID = hyperparams["ARCHITECTURE_ID"]
    EPOCHS = hyperparams["EPOCHS"]
    DIAGRAMS_DATA_PATH = hyperparams["DIAGRAMS_DATA_PATH"]
    WEIGHTS_PATH = hyperparams["WEIGHTS_PATH"]
    LEARNING_RATE = hyperparams["LEARNING_RATE"]
    LATENT_DIM = hyperparams["LATENT_DIM"]
    GAMMA = hyperparams["GAMMA"]
    KL_BETA = hyperparams["KL_BETA"]
    PATIENCE = hyperparams["PATIENCE"]
    DIAGRAMS_PATH = hyperparams["DIAGRAMS_PATH"]
    VISUALIZE_EVERY = hyperparams["VISUALIZE_EVERY"]
    GRADIENT_CLIP = hyperparams["GRADIENT_CLIP"]

    # Transform for grayscale: resize -> grayscale (3 channels, same values) -> tensor
    transform_grayscale = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ToGrayscale(num_output_channels=3),  # 3 channels with same grayscale values
        transforms.ToTensor()                 # (3, H, W)
    ])

    # Transform for green channel: resize -> extract green to 3-channel (only green populated)
    transform_green_channel = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ExtractGreenChannelTo3Channel()  # Creates (3, H, W) with [0, green, 0]
    ])

    train_dataset = PairedImageFolder(
        TRAIN_DATA_PATH,
        input_transform=transform_grayscale,      # Input: grayscale in 3 channels (3, H, W)
        target_transform=transform_green_channel  # Target: RGB with only green (3, H, W)
    )
    validation_dataset = PairedImageFolder(
        VALIDATION_DATA_PATH,
        input_transform=transform_grayscale,      # Input: grayscale in 3 channels (3, H, W)
        target_transform=transform_green_channel  # Target: RGB with only green (3, H, W)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )

    encoder = buildNeuralNetworkFromJson(
        f"./network_architectures/vae_encoder/{ARCHITECTURE_ID}.json"
    )
    decoder = buildNeuralNetworkFromJson(
        f"./network_architectures/vae_decoder/{ARCHITECTURE_ID}.json"
    )
    vae = VAE(
        encoder,
        decoder,
        latent_dimensions=LATENT_DIM,
        input_channels=INPUT_CHANNELS,
        input_image_size=IMAGE_RESIZE
    )
    vae = vae.to(device)

    models = {'vae': vae}

    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    optimizers = {'vae': optimizer}

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=GAMMA)
    schedulers = {'vae': scheduler}

    loss_fn = VAELoss(
        reconstruction_loss_fn=nn.MSELoss(),
        kl_weight=KL_BETA
    )

    metrics = VAEMetrics()

    result = train(
        models=models,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizers=optimizers,
        loss_fn=loss_fn,
        metrics=metrics,
        schedulers=schedulers,
        device=device,
        num_epochs=EPOCHS,
        diagrams_data_path=DIAGRAMS_DATA_PATH,
        hyperparams_id=HYPERPARAMS_ID,
        weights_path=WEIGHTS_PATH,
        diagrams_path=DIAGRAMS_PATH,
        launch_number=args.launch_number,
        visualize_every_xth_epoch=VISUALIZE_EVERY,
        max_patience=PATIENCE,
        model_type="VAE",
        gradient_clip=GRADIENT_CLIP,
        control_fn=vaeControlFunction,
        early_stopping_metric="total_loss"
    )

    print("\nTraining completed!")
    print("Task: VAE Grayscale (3ch, repeated) → Green Channel in RGB format (3ch)")
    print("Input format: [gray, gray, gray]")
    print("Target format: [0, green_values, 0]")
    print(f"Latent dim: {LATENT_DIM}, KL beta: {KL_BETA}")
    history = metrics.getHistoryLists()
    print(f"Final train loss: {history['train_total_losses'][-1]:.6f}")
    print(f"Final val loss: {history['val_total_losses'][-1]:.6f}")
    print(f"Best val loss: {min(history['val_total_losses']):.6f}")
    print(f"Total epochs trained: {result['epochs_trained']}")
