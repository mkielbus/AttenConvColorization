from pyaiwrap.config import buildNeuralNetworkFromJson
from pyaiwrap.train import train
from pyaiwrap.loss import GANLoss
from pyaiwrap.metrics import GANMetrics
from pyaiwrap.control import GANControlFunc
from pyaiwrap.gan import loadHyperparameters, warmupGAN
from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.transforms import ToGrayscale, ExtractGreenChannel
from pyaiwrap.utils import prepareDevice
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import torch.nn as nn
from torchvision.datasets import ImageFolder


def parseCMDArgs():
    """Parse command-line arguments for the program."""
    parser = argparse.ArgumentParser(description="Train GAN with configurable hyperparameters.")
    parser.add_argument(
        "--hyperparams",
        type=str,
        required=False,
        default="./hyperparams_gan/0.json",
        help="Path to the JSON file containing hyperparameters"
    )
    parser.add_argument(
        "--launch_number",
        type=int,
        required=False,
        default=0,
        help="Launch number for this training run"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=False,
        help="Do not use GAN warmup before adversarial training"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    device = prepareDevice()
    args = parseCMDArgs()

    # Load hyperparameters
    hyperparams = loadHyperparameters(args.hyperparams)
    BATCH_SIZE = hyperparams["BATCH_SIZE"]
    TRAIN_DATA_PATH = hyperparams["TRAIN_DATA_PATH"]
    VALIDATION_DATA_PATH = hyperparams["VALIDATION_DATA_PATH"]
    ARCHITECTURE_ID = hyperparams["ARCHITECTURE_ID"]
    HYPERPARAMS_ID = hyperparams["HYPERPARAMS_ID"]
    LEARNING_RATE = hyperparams["LEARNING_RATE"]
    GAMMA = hyperparams["GAMMA"]
    IMAGE_RESIZE = hyperparams["IMAGE_RESIZE"]
    IMAGE_CHANNELS = hyperparams["IMAGE_CHANNELS"]
    EPOCHS = hyperparams["EPOCHS"]
    DIAGRAMS_DATA_PATH = hyperparams["DIAGRAMS_DATA_PATH"]
    WEIGHTS_PATH = hyperparams["WEIGHTS_PATH"]
    PATIENCE = hyperparams["PATIENCE"]
    DIAGRAMS_PATH = hyperparams["DIAGRAMS_PATH"]
    VISUALIZE_EVERY = hyperparams["VISUALIZE_EVERY"]
    GRADIENT_CLIP = hyperparams["GRADIENT_CLIP"]
    WARMUP_EPOCHS = hyperparams["WARMUP_EPOCHS"]

    # Transform for grayscale: resize -> grayscale (1 channel) -> tensor
    transform_grayscale = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ToGrayscale(num_output_channels=1),
        transforms.ToTensor()                 # (1, H, W)
    ])

    # Transform for green channel: resize -> extract green to 1-channel
    transform_green_channel = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ExtractGreenChannel()  # Creates (1, H, W)
    ])

    train_dataset = PairedImageFolder(
        TRAIN_DATA_PATH,
        input_transform=transform_grayscale,
        target_transform=transform_green_channel
    )
    validation_dataset = PairedImageFolder(
        VALIDATION_DATA_PATH,
        input_transform=transform_grayscale,
        target_transform=transform_green_channel
    )

    warmup_dataset = ImageFolder(TRAIN_DATA_PATH, transform=transform_green_channel)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )

    warmup_loader = DataLoader(
        warmup_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )

    generator = buildNeuralNetworkFromJson(
        f"./network_architectures/gan_generator/{ARCHITECTURE_ID}.json"
    )
    generator = generator.to(device)

    discriminator = buildNeuralNetworkFromJson(
        f"./network_architectures/gan_discriminator/{ARCHITECTURE_ID}.json"
    )
    discriminator = discriminator.to(device)

    models = {
        'generator': generator,
        'discriminator': discriminator
    }

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    optimizers = {
        'generator': generator_optimizer,
        'discriminator': discriminator_optimizer
    }

    generator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=generator_optimizer, gamma=GAMMA)
    discriminator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=discriminator_optimizer, gamma=GAMMA)

    schedulers = {
        'generator': generator_scheduler,
        'discriminator': discriminator_scheduler
    }

    loss_fn = GANLoss(
        criterion=nn.MSELoss()
    )

    metrics = GANMetrics()

    if not args.no_warmup:
        warmupGAN(generator, discriminator, warmup_loader, generator_optimizer,
                  discriminator_optimizer, device, WARMUP_EPOCHS, GRADIENT_CLIP)

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
        model_type="GAN",
        gradient_clip=GRADIENT_CLIP,
        control_fn=GANControlFunc(target_channel="G"),
        early_stopping_metric="generator_loss"
    )

    print("\nTraining completed!")
    print("Task: GAN Grayscale (3ch) → Green Channel (3ch)")
    print("Input format: [gray, gray, gray]")
    print("Target format: [0, green_values, 0]")

    history = metrics.getHistoryLists()

    print(f"Final train G loss: {history['train_generator_loss'][-1]:.6f}")
    print(f"Final train D loss: {history['train_discriminator_loss'][-1]:.6f}")
    print(f"Final val G loss: {history['val_generator_loss'][-1]:.6f}")
    print(f"Final val D loss: {history['val_discriminator_loss'][-1]:.6f}")
    print(f"Total epochs trained: {result['epochs_trained']}")
