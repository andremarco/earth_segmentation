import matplotlib.pyplot as plt
import torch


def plot_img_and_mask(img, mask, dir_checkpoint):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    plt.title('Input')
    plt.imshow(img.permute(1, 2, 0).type("torch.IntTensor"))
    plt.savefig(dir_checkpoint + "true_image.png")

    pred = torch.sigmoid(mask)
    pred = (pred > 0.5).float()
    fig, ax = plt.subplots(3, 2)
    k1 = 0
    k2 = 0
    for i in range(classes):
        ax[k1,k2].set_title(f'Class {i+1}')
        ax[k1,k2].imshow(pred.permute(1, 2, 0)[:, :, i], cmap='gray')
        ax[k1, k2].set_xticks([]), ax[k1, k2].set_yticks([])
        if k1 < 2:
            k1 += 1
        else:
            k1 = 0
            k2 += 1
    plt.savefig(dir_checkpoint + "predicted_masks.png")
