import numpy as np
import torch
import random
import os
import matplotlib.cm as cm
import yaml
from omegaconf import OmegaConf
import glob



class RandomFlipPair:
    """
    Will randomly flip patches as they are passed over as augmentations. 
    In practice this will create multiple slightly different images from same patch
    """

    def __call__(self, sample):
        before, after, y = sample['before'], sample['after'], sample['y']

        if random.random() > 0.5:
            for m in before.keys():
                before[m] = torch.flip(before[m], dims=[2])
                after[m]  = torch.flip(after[m], dims=[2])
            y = torch.flip(y, dims=[1])

        if random.random() > 0.5:
            for m in before.keys():
                before[m] = torch.flip(before[m], dims=[1])
                after[m]  = torch.flip(after[m], dims=[1])
            y = torch.flip(y, dims=[0])

        return {'before': before, 'after': after, 'y': y}




class RandomRotationPair:
    """
    Will randomly rotate patches as they are passed over as augmentations. 
    In practice this will create multiple slightly different images from same patch.
    Rotation and flipping always ran together.
    """
    def __call__(self, sample):
        before, after, y = sample['before'], sample['after'], sample['y']
        k = random.randint(0, 3)

        for m in before.keys():
            before[m] = torch.rot90(before[m], k, dims=[1, 2])
            after[m]  = torch.rot90(after[m], k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[0, 1])

        return {'before': before, 'after': after, 'y': y}




def standardize(data: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    """
    Standardize the input x data so that backpropogation is stable
    """
    means = data.mean(dim=dim, keepdim=True)
    stds = data.std(dim=dim, keepdim=True)
    normalized = (data - means) / (stds + eps)
    return normalized


def weights(dataloader, num_classes=3, ignore_index=0, device="cpu"):
    """
    Weight classes in the loss function by their frequency
    """
    num_pixels = torch.zeros(num_classes, dtype=torch.float, device=device)

    for _, y in dataloader:   # y has shape [B, H, W]
        y = y.to(device)
        for c in range(num_classes):
            if c != ignore_index:
                num_pixels[c] += (y == c).sum()

    nonzero = num_pixels[num_pixels> 0]
    inv_freq = 1.0 / nonzero
    norm_inv_freq = inv_freq / inv_freq.mean()

    inverse = torch.zeros_like(num_pixels)
    inverse[num_pixels > 0] = norm_inv_freq

    # return num_pixels #On original run Evan used this, I recommend experimenting between values
    return inverse, num_pixels



def calc_batch_metrics(logits, y, ignore_index=None, positive_class=2, negative_class=1):
    """
    Calculate metrics by batch in train/validation loop. Later to be fed into calc_epoch_metrics
    """
    predictions = torch.argmax(logits, dim=1)

    if ignore_index is not None:
        mask = (y != ignore_index)
    else:
        mask = torch.ones_like(y, dtype=torch.bool)

    TP = ((predictions == positive_class) & (y == positive_class) & mask).sum().item()
    TN = ((predictions == negative_class) & (y == negative_class) & mask).sum().item()
    FP = ((predictions == positive_class) & (y == negative_class) & mask).sum().item()
    FN = ((predictions == negative_class) & (y == positive_class) & mask).sum().item()

    return TP, FP, FN, TN


def calc_epoch_metrics(TP, FP, FN, TN):
    """
    Calculate metrics by epoch in train/validation loop
    """
    # To avoid div/0 issue
    eps = .1e-6

    accuracy = (TP+TN) /(TP + FP + TN + FN + eps)
    precision = TP /(TP + FP + eps)
    recall = TP /(TP + FN + eps)
    f1 = (2*precision*recall) /(precision + recall  + eps)
    iou = TP/ (TP + FP + FN + eps)
    
    results = {"Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou}

    return results


def move_to_device(batch, device):
    """
    Helper function to pass x dict values to device
    """
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch


def add_batch_dim(batch):
    """
    Helper to prepend a batch dimension to (unbatched) tensors, e.g. when a
    DataLoader is configured with batch_size=None and yields one sample at a
    time without collation.
    """
    if torch.is_tensor(batch):
        return batch.unsqueeze(0)
    elif isinstance(batch, dict):
        return {k: add_batch_dim(v) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        return [add_batch_dim(v) for v in batch]
    else:
        return batch


# Set seeds
def set_seeds(seed: int=22):
    """
    Sets torch seeds for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def save_checkpoint(encoder, change_fusion, decoder, optimizer, epoch, val_loss, cfg,
                    save_dir="checkpoints", prefix="best"):
    """
    Save model checkpoint and config, deleting the previous best for this prefix.
    """
    os.makedirs(save_dir, exist_ok=True)

    old_checkpoints = glob.glob(os.path.join(save_dir, f"{prefix}_model_*.pt"))
    for f in old_checkpoints:
        os.remove(f)

    old_configs = glob.glob(os.path.join(save_dir, f"{prefix}_config_*.yaml"))
    for f in old_configs:
        os.remove(f)

    checkpoint_path = os.path.join(save_dir, f"{prefix}_model_epoch{epoch}_valloss{val_loss:.4f}.pt")
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'change_fusion_state_dict': change_fusion.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'val_loss': val_loss,
    }, checkpoint_path)

    cfg_path = os.path.join(save_dir, f"{prefix}_config_epoch{epoch}_valloss{val_loss:.4f}.yaml")
    OmegaConf.save(cfg, cfg_path)
    print(f"Saved new best checkpoint: {checkpoint_path}")
    return checkpoint_path


def calc_confusion_counts(pred, true, ignore_index=0, positive_class=2, negative_class=1):
    """
    Raw TP/FP/FN/TN pixel counts for one prediction/label tile pair.
    Shared building block for per-chip, per-event (pooled), and micro
    (globally pooled) metrics -- pooling raw counts before computing ratios
    is what makes those three consistent with each other.
    """
    pred = pred.long()
    true = true.long()

    if ignore_index is not None:
        mask = (true != ignore_index)
        pred = pred[mask]
        true = true[mask]

    TP = ((pred == positive_class) & (true == positive_class)).sum().item()
    TN = ((pred == negative_class) & (true == negative_class)).sum().item()
    FP = ((pred == positive_class) & (true == negative_class)).sum().item()
    FN = ((pred == negative_class) & (true == positive_class)).sum().item()
    return TP, FP, FN, TN


def calc_test_metrics(image_tiles_pred, image_tiles_true, ignore_index=0, positive_class=2, negative_class=1):
    """
    calculate metrics of interest for our test runs only
    """
    all_results = {}

    # run over each image tile (true & pred)
    for idx in image_tiles_pred.keys():
        TP, FP, FN, TN = calc_confusion_counts(
            image_tiles_pred[idx], image_tiles_true[idx],
            ignore_index=ignore_index, positive_class=positive_class, negative_class=negative_class,
        )
        all_results[idx] = calc_epoch_metrics(TP, FP, FN, TN)

    return all_results


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=None):
    """
    Percentile bootstrap CI over a list of per-group (e.g. per-event) metric
    values -- resamples the groups themselves (not pixels), which is the
    right unit given each event's metric is already a single pooled number.
    Returns (low, high). With <2 values, returns (nan, nan) since a CI over
    a single observation is meaningless.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[b] = sample.mean()

    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def tensor_to_color_image(tensor, num_classes=3):
    """
    This will help our images show up in color on tensorboard
    """
    if isinstance(tensor, list):  # handle list case
        tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] == 1:  # (1,H,W)
        tensor = tensor.squeeze(0)
    tensor_np = tensor.detach().cpu().numpy().astype(np.uint8)

    cmap = cm.get_cmap('tab10', num_classes)
    color_np = (cmap(tensor_np)[:, :, :3] * 255).astype(np.uint8)
    color_tensor = torch.from_numpy(color_np).permute(2, 0, 1)  # (3,H,W)
    return color_tensor
