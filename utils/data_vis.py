import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/andreabp/PycharmProjects/earth_segmentation'])

import argparse
import matplotlib.pyplot as plt
import torch
import tifffile as tiff

from PIL import Image
from icnet import ICNet
from unet import UNet
from utils.dataset import BasicDataset


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
        ax[k1, k2].set_title(f'Class {i+1}')
        ax[k1, k2].imshow(pred.permute(1, 2, 0)[:, :, i], cmap='gray')
        ax[k1, k2].set_xticks([]), ax[k1, k2].set_yticks([])
        if k1 < 2:
            k1 += 1
        else:
            k1 = 0
            k2 += 1
    plt.savefig(dir_checkpoint + "predicted_masks.png")


def plot_imgs_pred():
    """
    Funzione
    :return:
    """
    args = get_plot_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # img = Image.open(args.dir_img)
    img = tiff.imread(args.dir_img)
    img = BasicDataset.preprocess(img, scale=args.scale)
    img = torch.from_numpy(img).type(torch.FloatTensor)

    if args.model_arch == 'unet':
        net = UNet(n_channels=4, n_classes=4, bilinear=True)

    elif args.model_arch == 'icnet':
        net = ICNet(n_channels=4, n_classes=4, pretrained_base=False)

    net.load_state_dict(torch.load(args.checkpoint_net, map_location=device))
    net.to(device=device)
    net.eval()

    img = img.to(device=device, dtype=torch.float32)
    img = img.unsqueeze(0)

    with torch.no_grad():
        if args.model_arch == 'icnet':
            mask_pred, pred_sub4, pred_sub8, pred_sub16 = net(img)
        else:
            mask_pred = net(img)

    plt.imshow(img[0][0])
    plt.colorbar()
    plt.savefig(args.dir_output + "original_img.png")
    plt.clf()

    for i, c in enumerate(mask_pred):
        n_classes = c.size(0)
        classes = range(n_classes)
        c = torch.sigmoid(c)
        max_index = torch.max(c, 0).indices
        for class_index in classes:
            # Vediamo la predizione
            jaccard_input = (max_index == class_index).float()
            plt.imshow(jaccard_input)
            plt.colorbar()
            plt.savefig(args.dir_output + f"pred_cls_{class_index}.png")
            plt.clf()


def get_plot_args():
    parser = argparse.ArgumentParser(description='Plot predicted classes',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--model_arch", help="Model architecture", type=str,
                        default='unet')
    parser.add_argument("-i", "--dir_img", help="Images directory", type=str,
                        default='./data/imgs/img_prova.tif')
    parser.add_argument("-o", "--dir_output", help="Output directory", type=str,
                        default='./data/plot/', dest="dir_output")
    parser.add_argument("-c", "--checkpoint_net", help="Checkpoint directory, where to load", type=str,
                        default='model.pth')
    parser.add_argument('-k', '--input_channels', dest='input_channels', type=int, default=4,
                        help='Number of input channels')
    parser.add_argument('-w', '--output_channels', dest='output_channels', type=int, default=4,
                        help='Number of output channels')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    return parser.parse_args()


if __name__ == '__main__':
    plot_imgs_pred()
