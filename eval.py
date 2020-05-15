import torch
import torch.nn.functional as F
from tqdm import tqdm

from iou_loss import iou_coeff
from utils.data_vis import plot_img_and_mask


def eval_net(net, loader, device, plot=False):
    """Evaluation with the iou coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot_ce = 0
    tot_iou = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            tot_ce += F.cross_entropy(mask_pred, true_masks.squeeze(1)).item()
            tot_iou += iou_coeff(mask_pred, true_masks.squeeze(1)).item()

            pbar.update()

            if plot:
                for i, c in enumerate(zip(imgs, mask_pred)):
                    plot_img_and_mask(c[0], c[1])

    return tot_ce / n_val, tot_iou / n_val
