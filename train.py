import argparse
import logging
import os
import sys
import socket
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from icnet import ICNet, ICNetLoss, IterationPolyLR

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


def train_net(dir_img,
              dir_mask,
              dir_checkpoint,
              net,
              device,
              model_arch,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              test_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_test = int(len(dataset) * test_percent)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val - n_test
    train, val, test = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train, batch_size=min(n_train, batch_size), shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val, batch_size=min(n_val, batch_size), shuffle=False, pin_memory=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test, batch_size=min(n_test, batch_size), shuffle=False, pin_memory=True, drop_last=True, num_workers=8)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(dir_checkpoint, current_time + '_' + socket.gethostname() + f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Testing size:    {n_test}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    if model_arch == 'icnet':
        optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        scheduler = IterationPolyLR(optimizer, max_iters=epochs*len(train_loader), power=0.9)
        criterion = ICNetLoss()
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
        criterion = nn.CrossEntropyLoss()

    # Training
    early_stopping = 10
    epochs_no_best = 0
    best_val_iou = 0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks.squeeze(1))
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

            writer.add_scalar('Loss/train', epoch_loss, epoch)
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                if value.grad is None:
                    continue
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)

        # Validation
        val_ce, val_iou = eval_net(net, model_arch, val_loader, device)
        scheduler.step(val_ce)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        logging.info('Validation cross entropy: {}'.format(val_ce))
        logging.info('Validation IoU: {}'.format(val_iou))
        writer.add_scalar('Loss/test', val_ce, epoch)
        writer.add_scalar('IoU/test', val_iou, epoch)

        # writer.add_images('images', imgs, epoch)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            if val_iou > best_val_iou:
                best_epoch = epoch
                torch.save(net.state_dict(), dir_checkpoint + 'model.pth')
                logging.info('Best model saved !')

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_best = 0
        else:
            epochs_no_best += 1
            if epochs_no_best > early_stopping:
                logging.info('Stopped for Early Stopping')
                break

    logging.info('Best Validation IoU Coeff: {}'.format(best_val_iou))
    logging.info(f'Best Epoch: {best_epoch + 1}')
    writer.close()

    # Testing
    net.load_state_dict(torch.load(dir_checkpoint + 'model.pth', map_location=device))
    test_ce, test_iou = eval_net(net, model_arch, test_loader, device, plot=False, dir_checkpoint=dir_checkpoint)
    logging.info('Test cross entropy: {}'.format(test_ce))
    logging.info('Test IoU: {}'.format(test_iou))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--model_arch", help="Model architecture", type=str,
                        default='unet')
    parser.add_argument("-p", "--pretrained", help="Set True if you want to initialize the model with pretrained prameters", type=str,
                        default=False)
    parser.add_argument("-i", "--dir_img", help="Images directory", type=str,
                        default='./data/rgb/')
    parser.add_argument("-m", "--dir_mask", help="Masks directory", type=str,
                        default='./data/mask/')
    parser.add_argument("-c", "--dir_checkpoint", help="Checkpoint directory, where to save", type=str,
                        default='./checkpoints/')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-ic', '--input_channels', dest='input_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('-oc', '--output_channels', dest='output_channels', type=int, default=4,
                        help='Number of output channels')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--test', dest='test', type=float, default=10.0,
                        help='Percent of the data that is used as test (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    models = "'icnet', 'unet'"
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    dir_img = args.dir_img
    dir_mask = args.dir_mask
    dir_checkpoint = args.dir_checkpoint

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    model_arch = args.model_arch
    if args.pretrained != False:
        args.pretrained = True
        print('Initializing pretrained model...')
    n_channels_input = args.input_channels
    n_channels_output = args.output_channels
    if model_arch == 'unet':
        net = UNet(n_channels=n_channels_input, n_classes=n_channels_output, bilinear=True)
        logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    elif model_arch == 'icnet':
        net = ICNet(n_channels=n_channels_input, n_classes=n_channels_output, pretrained_base=args.pretrained)
        logging.info(f'Network:\n'
                    f'\t{net.n_channels} input channels\n'
                    f'\t{net.n_classes} output channels (classes)')
    else:
        print(f'Model not implemented yet. Please choose one of the available models: {models}')
        exit(0)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(dir_img=dir_img,
                  dir_mask=dir_mask,
                  dir_checkpoint=dir_checkpoint,
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  test_percent=args.test / 100,
                  model_arch=model_arch)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
