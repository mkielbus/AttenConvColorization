from pyaiwrap.train import train
from pyaiwrap.config import buildNeuralNetworkFromJson
from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.loss import GeneratorColorizationLoss
from pyaiwrap.metrics import GeneratorColorizationMetrics
from pyaiwrap.control import GeneratorControlFunc
from pyaiwrap.generator import loadHyperparameters
from pyaiwrap.transforms import ToGrayscale, ExtractRedChannel, ExtractGreenChannel, \
     ExtractBlueChannel
from pyaiwrap.neural_network import ConvAttenColorizationNetwork
from pyaiwrap.utils import prepareDevice
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from typing import Tuple
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def getTargetChannelTransform(target_channel: str, image_size: int, has_submodules: bool) -> Tuple[transforms.Compose,
                                                                                                   str]:
    """
    Get the appropriate transform for the target color channel.

    Args:
        target_channel: Channel to extract ("R", "G", "B", or "RGB")
        image_size: Size to resize images to
        has_submodules: Whether SUBMODULES is not empty

    Returns:
        Tuple of (transform, channel_format_string)

    Raises:
        ValueError: If target_channel is not "R", "G", "B", or "RGB"
    """
    if has_submodules:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        channel_format = "[red_values, green_values, blue_values]"
    elif target_channel == "R":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractRedChannel(num_output_channels=1)  # [1, H, W]
        ])
        channel_format = "[red_values] (single channel)"
    elif target_channel == "G":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractGreenChannel(num_output_channels=1)  # [1, H, W]
        ])
        channel_format = "[green_values] (single channel)"
    elif target_channel == "B":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractBlueChannel(num_output_channels=1)  # [1, H, W]
        ])
        channel_format = "[blue_values] (single channel)"
    else:
        raise ValueError(f"TARGET_CHANNEL must be 'R', 'G', or 'B' when no submodules, got '{target_channel}'")

    return transform, channel_format


def parseCMDArgs():
    parser = argparse.ArgumentParser(description="Train generator model with configurable hyperparameters.")
    parser.add_argument(
        "--hyperparams",
        type=str,
        required=False,
        default="./hyperparams_generator/0.json",
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
    SUBMODULES = hyperparams.get("SUBMODULES", {})
    EPOCHS = hyperparams["EPOCHS"]
    DIAGRAMS_DATA_PATH = hyperparams["DIAGRAMS_DATA_PATH"]
    WEIGHTS_PATH = hyperparams["WEIGHTS_PATH"]
    LEARNING_RATE = hyperparams["LEARNING_RATE"]
    GAMMA = hyperparams["GAMMA"]
    PATIENCE = hyperparams["PATIENCE"]
    DIAGRAMS_PATH = hyperparams["DIAGRAMS_PATH"]
    VISUALIZE_EVERY = hyperparams["VISUALIZE_EVERY"]
    GRADIENT_CLIP = hyperparams["GRADIENT_CLIP"]
    PERCEPTUAL_WEIGHT = hyperparams["PERCEPTUAL_WEIGHT"]
    COLORFULNESS_WEIGHT = hyperparams["COLORFULNESS_WEIGHT"]
    COLORFULNESS_TARGET = hyperparams["COLORFULNESS_TARGET"]
    USE_LPIPS = hyperparams["USE_LPIPS"]
    LPIPS_NET = hyperparams["LPIPS_NET"]
    TARGET_CHANNEL = hyperparams["TARGET_CHANNEL"]

    has_submodules = bool(SUBMODULES)

    print("Training Configuration")
    print(f"Hyperparams ID: {HYPERPARAMS_ID}")
    print(f"Architecture ID: {ARCHITECTURE_ID}")
    print(f"Launch Number: {args.launch_number}")
    print(f"Target Channel: {TARGET_CHANNEL}")
    print(f"Has Submodules: {has_submodules}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_RESIZE}")
    print(f"Input Channels: {INPUT_CHANNELS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print(f"Gradient Clip: {GRADIENT_CLIP}")
    print("\nLoss Configuration:")
    print(f"  Perceptual Weight: {PERCEPTUAL_WEIGHT}")
    print(f"  Use LPIPS: {USE_LPIPS}")
    if USE_LPIPS:
        print(f"  LPIPS Network: {LPIPS_NET}")
    print(f"  Colorfulness Weight: {COLORFULNESS_WEIGHT}")
    if COLORFULNESS_TARGET is not None:
        print(f"  Colorfulness Target: {COLORFULNESS_TARGET}")
    else:
        print("  Colorfulness Target: Match Original")

    transform_luminance = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        ToGrayscale(num_output_channels=INPUT_CHANNELS),
        transforms.ToTensor()
    ])

    transform_target_channel, channel_format = getTargetChannelTransform(TARGET_CHANNEL, IMAGE_RESIZE, has_submodules)

    train_dataset = PairedImageFolder(
        TRAIN_DATA_PATH,
        input_transform=transform_luminance,
        target_transform=transform_target_channel
    )
    validation_dataset = PairedImageFolder(
        VALIDATION_DATA_PATH,
        input_transform=transform_luminance,
        target_transform=transform_target_channel
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )

    print("Building generator model...")
    if not SUBMODULES:
        generator = buildNeuralNetworkFromJson(
            f"./network_architectures/generators/{ARCHITECTURE_ID}.json"
        )
    else:
        generator = ConvAttenColorizationNetwork(
            architecture_path=f"./network_architectures/generators/{ARCHITECTURE_ID}.json",
            pretrained_models_config=SUBMODULES
        )
    generator = generator.to(device)

    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    models = {'generator': generator}

    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizers = {'generator': optimizer}

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=GAMMA)
    schedulers = {'generator': scheduler}

    loss_fn = GeneratorColorizationLoss(
        reconstruction_loss_fn=nn.MSELoss(),
        perceptual_weight=PERCEPTUAL_WEIGHT,
        colorfulness_weight=COLORFULNESS_WEIGHT,
        colorfulness_target=COLORFULNESS_TARGET,
        use_lpips=USE_LPIPS,
        lpips_net=LPIPS_NET,
        device=device
    )

    metrics = GeneratorColorizationMetrics(use_colorfulness=COLORFULNESS_WEIGHT > 0,
                                           use_perceptual_loss=PERCEPTUAL_WEIGHT > 0)

    print("Starting training...\n")
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
        model_type="custom",
        gradient_clip=GRADIENT_CLIP,
        control_fn=GeneratorControlFunc(target_channel=TARGET_CHANNEL),
        early_stopping_metric="total_loss"
    )

    print("Training Completed!")
    print(f"Task: Luminance (1ch) → {TARGET_CHANNEL} Channel (3ch) Colorization")
    print("Input format: [luminance]")
    print(f"Target format: {channel_format}")

    history = metrics.getHistoryLists()

    print("\nFinal Metrics:")
    print(f"  Train Total Loss: {history['train_total_loss'][-1]:.6f}")
    print(f"  Val Total Loss: {history['val_total_loss'][-1]:.6f}")
    print(f"  Best Val Loss: {min(history['val_total_loss']):.6f}")

    if PERCEPTUAL_WEIGHT > 0:
        print("\nPerceptual Loss:")
        print(f"  Train: {history['train_perceptual_loss'][-1]:.6f}")
        print(f"  Val: {history['val_perceptual_loss'][-1]:.6f}")

    if COLORFULNESS_WEIGHT > 0:
        print("\nColorfulness Metrics:")
        print(f"  Train Colorfulness Loss: {history['train_colorfulness_loss'][-1]:.6f}")
        print(f"  Val Colorfulness Loss: {history['val_colorfulness_loss'][-1]:.6f}")
        print(f"  Final Reconstructed: {history['val_colorfulness_recon'][-1]:.2f}")
        print(f"  Final Original: {history['val_colorfulness_original'][-1]:.2f}")

    print(f"\nTotal epochs trained: {result['epochs_trained']}")
    print(f"Early stopping triggered: {result.get('early_stopped', False)}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"Model saved to: {WEIGHTS_PATH}")
    print(f"Metrics saved to: {DIAGRAMS_DATA_PATH}")
    print(f"Visualizations saved to: {DIAGRAMS_PATH}")
