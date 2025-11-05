from pyaiwrap.train import train
from pyaiwrap.config import buildNeuralNetworkFromJson
from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.loss import GeneratorLoss
from pyaiwrap.metrics import GeneratorMetrics
from pyaiwrap.control import generatorControlFunction
from pyaiwrap.generator import loadHyperparameters
from pyaiwrap.transforms import ToGrayscale, ExtractGreenChannelTo3Channel
from pyaiwrap.utils import prepareDevice
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse


def parseCMDArgs():
    parser = argparse.ArgumentParser(description="Train generator model with configurable hyperparameters.")
    parser.add_argument(
        "--hyperparams",
        type=str,
        required=False,
        default="./hyperparams_generator/0.json",
        help="Path to the JSON file containing hyperparameters"
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
    GAMMA = hyperparams["GAMMA"]
    PATIENCE = hyperparams["PATIENCE"]
    DIAGRAMS_PATH = hyperparams["DIAGRAMS_PATH"]
    VISUALIZE_EVERY = hyperparams["VISUALIZE_EVERY"]

    # Transform for grayscale: resize -> grayscale (3 channels, same values) -> tensor
    transform_grayscale = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ToGrayscale(num_output_channels=3),  # 3 channels with same grayscale values
        transforms.ToTensor()                 # Convert to tensor: (3, H, W)
    ])

    # Transform for green channel: resize -> extract green to 3-channel (only green populated)
    transform_green_channel = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ExtractGreenChannelTo3Channel()  # Creates (3, H, W) with [0, green, 0]
    ])

    train_dataset = PairedImageFolder(
        TRAIN_DATA_PATH,
        input_transform=transform_grayscale,      # Input: grayscale (3, H, W)
        target_transform=transform_green_channel  # Target: green channel (3, H, W)
    )
    validation_dataset = PairedImageFolder(
        VALIDATION_DATA_PATH,
        input_transform=transform_grayscale,      # Input: grayscale (3, H, W)
        target_transform=transform_green_channel  # Target: green channel (3, H, W)
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

    generator = buildNeuralNetworkFromJson(
        f"./network_architectures/generators/{ARCHITECTURE_ID}.json"
    )
    generator = generator.to(device)

    models = {'generator': generator}

    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizers = {'generator': optimizer}

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=GAMMA)
    schedulers = {'generator': scheduler}

    loss_fn = GeneratorLoss(reconstruction_loss_fn=nn.MSELoss(), use_lpips=True, perceptual_weight=1.0, device=device)

    metrics = GeneratorMetrics()

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
        launch_number=0,
        visualize_every_xth_epoch=VISUALIZE_EVERY,
        max_patience=PATIENCE,
        model_type="ViT_gray_to_green",
        gradient_clip=1.0,
        control_fn=generatorControlFunction,
        early_stopping_metric="total_loss"
    )

    print("\nTraining completed!")
    print("Task: Grayscale (1ch) → Green Channel (1ch) Reconstruction")
    history = metrics.getHistoryLists()
    print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    print(f"Final val loss: {history['val_losses'][-1]:.6f}")
    print(f"Best val loss: {min(history['val_losses']):.6f}")
    print(f"Total epochs trained: {result['epochs_trained']}")
