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
    Use distinct prefixes per curriculum stage ("best_flood" vs "best") so Stage 1
    checkpoints are not deleted when Stage 2 saves a new best.
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


def calc_test_metrics(image_tiles_pred, image_tiles_true, ignore_index=0, positive_class=2, negative_class=1):
    """
    calculate metrics of interest for our test runs only
    """
    
    # To avoid div/0 issue
    eps = 1e-6

    all_results = {}

    # run over each image tile (true & pred)
    for idx in image_tiles_pred.keys():
        pred = image_tiles_pred[idx].long()
        true = image_tiles_true[idx].long()

        # Apply ignore mask (background)
        if ignore_index is not None:
            mask = (true != ignore_index)
            pred = pred[mask]
            true = true[mask]

        TP = ((pred == positive_class) & (true == positive_class)).sum().item()
        TN = ((pred == negative_class) & (true == negative_class)).sum().item()
        FP = ((pred == positive_class) & (true == negative_class)).sum().item()
        FN = ((pred == negative_class) & (true == positive_class)).sum().item()

        accuracy = (TP + TN) / (TP + FP + TN + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        iou = TP / (TP + FP + FN + eps)

        # Return all desired results
        all_results[idx] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou
        }

    return all_results


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
