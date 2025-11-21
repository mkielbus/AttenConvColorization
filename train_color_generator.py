from pyaiwrap.train import train
from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.loss import GeneratorColorizationLoss
from pyaiwrap.metrics import GeneratorColorizationMetrics
from pyaiwrap.control import GeneratorControlFunc
from pyaiwrap.config import loadConfig
from pyaiwrap.transforms import ChannelTransformCreator
from pyaiwrap.optimizers import createOptimizer
from pyaiwrap.schedulers import createScheduler
from pyaiwrap.utils import prepareDevice
from pyaiwrap.generator import createGenerator
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parseCMDArgs():
    parser = argparse.ArgumentParser(description="Train generator model with configurable hyperparameters.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./hyperparams_generator/0.json",
        help="Path to the JSON file containing configuration for training."
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


def printTrainingConfiguration(config, launch_number):
    """Print the training configuration."""
    has_submodules = bool(config["SUBMODULES"])

    print("Training Configuration")
    print(f"Hyperparams ID: {config['HYPERPARAMS_ID']}")
    print(f"Architecture ID: {config['ARCHITECTURE_ID']}")
    print(f"Launch Number: {launch_number}")
    print(f"Target Channel: {config['TARGET_CHANNEL']}")
    print(f"Has Submodules: {has_submodules}")
    print(f"Batch Size: {config['BATCH_SIZE']}")
    print(f"Image Size: {config['IMAGE_RESIZE']}")
    print(f"Input Channel: {config['INPUT_CHANNEL']}")
    print(f"Learning Rate: {config['LEARNING_RATE']}")
    print(f"Epochs: {config['EPOCHS']}")
    print(f"Patience: {config['PATIENCE']}")
    print(f"Gradient Clip: {config['GRADIENT_CLIP']}")
    print(f"Weight Decay: {config['WEIGHT_DECAY']}")
    print(f"Optimizer type: {config['OPTIMIZER_TYPE']}")
    print(f"Scheduler type: {config['SCHEDULER_TYPE']}")
    print("\nLoss Configuration:")
    if "perceptual" in config['LOSS_TYPES']:
        print(f"  Perceptual Weight: {config['PERCEPTUAL_WEIGHT']}")
        print(f"  Use LPIPS: {config['USE_LPIPS']}")
        print(f"  LPIPS Network: {config['LPIPS_NET']}")
    if "colorfulness" in config['LOSS_TYPES']:
        print(f"  Colorfulness Weight: {config['COLORFULNESS_WEIGHT']}")
        if config['COLORFULNESS_TARGET'] is not None:
            print(f"  Colorfulness Target: {config['COLORFULNESS_TARGET']}")
        else:
            print("  Colorfulness Target: Match Original")


def createDataLoaders(config):
    """Create train and validation data loaders."""
    transform_input = ChannelTransformCreator.getTransform(
        config["INPUT_CHANNEL"], config["IMAGE_RESIZE"], config["OUTPUT_CHANNELS"], is_input=True
    )
    transform_target = ChannelTransformCreator.getTransform(
        config["TARGET_CHANNEL"], config["IMAGE_RESIZE"], config["TARGET_OUTPUT_CHANNELS"], is_input=False
    )

    train_dataset = PairedImageFolder(
        config["TRAIN_DATA_PATH"],
        input_transform=transform_input,
        target_transform=transform_target
    )
    validation_dataset = PairedImageFolder(
        config["VALIDATION_DATA_PATH"],
        input_transform=transform_input,
        target_transform=transform_target
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=8
    )

    return train_loader, validation_loader, len(train_dataset), len(validation_dataset)


def createGeneratorModel(config, device):
    """Create and return the generator model."""
    generator = createGenerator(config=config, device=device)
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)

    return generator, total_params, trainable_params


def prepareOptimizer(model_parameters, config):
    """Create optimizer."""
    optimizer = createOptimizer(model_parameters, config)
    return optimizer


def prepareScheduler(optimizer, config, train_loader_len):
    """Create scheduler."""
    scheduler = createScheduler(
        optimizer=optimizer,
        config=config,
        train_loader_len=train_loader_len
    )
    return scheduler


def createLossFunction(config, device):
    """Create the loss function."""
    loss_fn = GeneratorColorizationLoss(
        reconstruction_loss_fn=nn.L1Loss(),
        recon_weight=config["RECON_WEIGHT"],
        perceptual_weight=config.get("PERCEPTUAL_WEIGHT", 0.0),
        colorfulness_weight=config.get("COLORFULNESS_WEIGHT", 0.0),
        colorfulness_target=config.get("COLORFULNESS_TARGET", 0.0),
        use_lpips=config.get("USE_LPIPS", False),
        lpips_net=config.get("LPIPS_NET", "alex"),
        device=device,
        target_channel=config["TARGET_CHANNEL"],
        input_channel=config["INPUT_CHANNEL"]
    )
    return loss_fn


def createMetrics(config):
    """Create metrics object."""
    metrics = GeneratorColorizationMetrics(
        use_colorfulness=config.get("COLORFULNESS_WEIGHT", 0.0) > 0,
        use_perceptual_loss=config.get("PERCEPTUAL_WEIGHT", 0.0) > 0
    )
    return metrics


def printFinalResults(result, metrics, config, optimizer):
    """Print final training results."""
    history = metrics.getHistoryLists()

    print("Training Completed!")
    print(f"Input format: {config['INPUT_CHANNEL']} - {config['OUTPUT_CHANNELS']} channels")
    print(f"Target format: {config['TARGET_CHANNEL']} - {config['TARGET_OUTPUT_CHANNELS']} channels")

    print("\nFinal Metrics:")
    print(f"  Train Total Loss: {history['train_total_loss'][-1]:.6f}")
    print(f"  Val Total Loss: {history['val_total_loss'][-1]:.6f}")
    print(f"  Best Val Loss: {min(history['val_total_loss']):.6f}")

    if config["PERCEPTUAL_WEIGHT"] > 0:
        print("\nPerceptual Loss:")
        print(f"  Train: {history['train_perceptual_loss'][-1]:.6f}")
        print(f"  Val: {history['val_perceptual_loss'][-1]:.6f}")

    if config["COLORFULNESS_WEIGHT"] > 0:
        print("\nColorfulness Metrics:")
        print(f"  Train Colorfulness Loss: {history['train_colorfulness_loss'][-1]:.6f}")
        print(f"  Val Colorfulness Loss: {history['val_colorfulness_loss'][-1]:.6f}")
        print(f"  Final Reconstructed: {history['val_colorfulness_recon'][-1]:.2f}")

    print(f"\nTotal epochs trained: {result['epochs_trained']}")
    print(f"Early stopping triggered: {result.get('early_stopped', False)}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"Model saved to: {config['WEIGHTS_PATH']}")
    print(f"Metrics saved to: {config['DIAGRAMS_DATA_PATH']}")
    print(f"Visualizations saved to: {config['DIAGRAMS_PATH']}")


if __name__ == "__main__":
    device = prepareDevice()
    args = parseCMDArgs()

    config = loadConfig(args.config)

    printTrainingConfiguration(config, args.launch_number)

    train_loader, validation_loader, train_samples, val_samples = createDataLoaders(config)
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}\n")

    print("Building generator model...")
    generator, total_params, trainable_params = createGeneratorModel(config, device)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    optimizer = prepareOptimizer(generator.parameters(), config)
    optimizers = {'generator': optimizer}

    scheduler = prepareScheduler(optimizer, config, len(train_loader))
    schedulers = {'generator': scheduler}

    models = {'generator': generator}

    loss_fn = createLossFunction(config, device)
    metrics = createMetrics(config)

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
        num_epochs=config["EPOCHS"],
        diagrams_data_path=config["DIAGRAMS_DATA_PATH"],
        hyperparams_id=config["HYPERPARAMS_ID"],
        weights_path=config["WEIGHTS_PATH"],
        diagrams_path=config["DIAGRAMS_PATH"],
        launch_number=args.launch_number,
        visualize_every_xth_epoch=config["VISUALIZE_EVERY"],
        max_patience=config["PATIENCE"],
        model_type="custom",
        gradient_clip=config["GRADIENT_CLIP"],
        control_fn=GeneratorControlFunc(
            target_channel=config["TARGET_CHANNEL"],
            input_channel=config["INPUT_CHANNEL"]
        ),
        early_stopping_metric="total_loss"
    )

    printFinalResults(result, metrics, config, optimizers['generator'])
