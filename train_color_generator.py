from pyaiwrap.train import train
from pyaiwrap.config import buildNeuralNetworkFromJson
from pyaiwrap.datasets import PairedImageFolder
from pyaiwrap.loss import GeneratorColorizationLoss
from pyaiwrap.metrics import GeneratorColorizationMetrics
from pyaiwrap.control import GeneratorControlFunc
from pyaiwrap.generator import loadHyperparameters
from pyaiwrap.transforms import channelTransform
from pyaiwrap.schedulers import createScheduler
from pyaiwrap.neural_network import ConvAttenColorizationNetwork
from pyaiwrap.utils import prepareDevice
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


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
    INPUT_CHANNEL = hyperparams["INPUT_CHANNEL"]
    OUTPUT_CHANNELS = hyperparams["OUTPUT_CHANNELS"]
    HYPERPARAMS_ID = hyperparams["HYPERPARAMS_ID"]
    ARCHITECTURE_ID = hyperparams["ARCHITECTURE_ID"]
    SUBMODULES = hyperparams.get("SUBMODULES", {})
    EPOCHS = hyperparams["EPOCHS"]
    DIAGRAMS_DATA_PATH = hyperparams["DIAGRAMS_DATA_PATH"]
    WEIGHTS_PATH = hyperparams["WEIGHTS_PATH"]
    LEARNING_RATE = hyperparams["LEARNING_RATE"]
    PATIENCE = hyperparams["PATIENCE"]
    DIAGRAMS_PATH = hyperparams["DIAGRAMS_PATH"]
    VISUALIZE_EVERY = hyperparams["VISUALIZE_EVERY"]
    GRADIENT_CLIP = hyperparams["GRADIENT_CLIP"]
    RECON_WEIGHT = hyperparams["RECON_WEIGHT"]
    PERCEPTUAL_WEIGHT = hyperparams["PERCEPTUAL_WEIGHT"]
    COLORFULNESS_WEIGHT = hyperparams["COLORFULNESS_WEIGHT"]
    COLORFULNESS_TARGET = hyperparams["COLORFULNESS_TARGET"]
    USE_LPIPS = hyperparams["USE_LPIPS"]
    LPIPS_NET = hyperparams["LPIPS_NET"]
    TARGET_CHANNEL = hyperparams["TARGET_CHANNEL"]
    TARGET_OUTPUT_CHANNELS = hyperparams["TARGET_OUTPUT_CHANNELS"]
    WEIGHT_DECAY = hyperparams.get("WEIGHT_DECAY", 0.01)
    USE_ADAMW = hyperparams.get("USE_ADAMW", True)
    B1 = hyperparams["B1"]
    B2 = hyperparams["B2"]

    has_submodules = bool(SUBMODULES)

    print("Training Configuration")
    print(f"Hyperparams ID: {HYPERPARAMS_ID}")
    print(f"Architecture ID: {ARCHITECTURE_ID}")
    print(f"Launch Number: {args.launch_number}")
    print(f"Target Channel: {TARGET_CHANNEL}")
    print(f"Has Submodules: {has_submodules}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_RESIZE}")
    print(f"Input Channel: {INPUT_CHANNEL}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print(f"Gradient Clip: {GRADIENT_CLIP}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Use AdamW: {USE_ADAMW}")
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

    transform_input = channelTransform(INPUT_CHANNEL, IMAGE_RESIZE, OUTPUT_CHANNELS, is_input=True)
    transform_target_channel = channelTransform(TARGET_CHANNEL, IMAGE_RESIZE, TARGET_OUTPUT_CHANNELS, is_input=False)

    train_dataset = PairedImageFolder(
        TRAIN_DATA_PATH,
        input_transform=transform_input,
        target_transform=transform_target_channel
    )
    validation_dataset = PairedImageFolder(
        VALIDATION_DATA_PATH,
        input_transform=transform_input,
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
        trainable_network = buildNeuralNetworkFromJson(
            f"./network_architectures/generators/{ARCHITECTURE_ID}.json"
        )

        generator = ConvAttenColorizationNetwork(
            pretrained_models_config=SUBMODULES,
            trainable_network=trainable_network
        )
    generator = generator.to(device)

    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    models = {'generator': generator}

    if USE_ADAMW:
        optimizer = torch.optim.AdamW(
            generator.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(B1, B2)
        )
        print(f"Using AdamW optimizer with weight decay: {WEIGHT_DECAY}")
    else:
        optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
        print("Using Adam optimizer")

    optimizers = {'generator': optimizer}

    scheduler = createScheduler(
        optimizer=optimizer,
        hyperparams=hyperparams,
        train_loader_len=len(train_loader),
        epochs=EPOCHS
    )
    schedulers = {'generator': scheduler}

    loss_fn = GeneratorColorizationLoss(
        reconstruction_loss_fn=nn.L1Loss(),
        recon_weight=RECON_WEIGHT,
        perceptual_weight=PERCEPTUAL_WEIGHT,
        colorfulness_weight=COLORFULNESS_WEIGHT,
        colorfulness_target=COLORFULNESS_TARGET,
        use_lpips=USE_LPIPS,
        lpips_net=LPIPS_NET,
        device=device,
        target_channel=TARGET_CHANNEL,
        input_channel=INPUT_CHANNEL
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
        control_fn=GeneratorControlFunc(target_channel=TARGET_CHANNEL, input_channel=INPUT_CHANNEL),
        early_stopping_metric="total_loss"
    )

    print("Training Completed!")
    print(f"Input format: {INPUT_CHANNEL} - {OUTPUT_CHANNELS} channels")
    print(f"Target format: {TARGET_CHANNEL} - {TARGET_OUTPUT_CHANNELS} channels")

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

    print(f"\nTotal epochs trained: {result['epochs_trained']}")
    print(f"Early stopping triggered: {result.get('early_stopped', False)}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"Model saved to: {WEIGHTS_PATH}")
    print(f"Metrics saved to: {DIAGRAMS_DATA_PATH}")
    print(f"Visualizations saved to: {DIAGRAMS_PATH}")
