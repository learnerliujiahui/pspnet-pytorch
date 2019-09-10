import os
import logging
import argparse
import timeit
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders import ISPRS
from dataloaders import custom_transforms as tr
from tensorboardX import SummaryWriter

from models.pspnet import PSPNet
import models.modules
import utils
import metrics

parser = argparse.ArgumentParser(description="Pytorch PSPNet parameters")
parser.add_argument('--data', type=str, default="ISPRS", help='Path to dataset folder')
parser.add_argument('--snapshot', type=str,
                    help='Path to pretrained weights')
parser.add_argument('--save_path', type=str, default="/home/f517/PycharmProjects/ISPRS-pspnet-pytorch/model",
                    help='Path for storing model snapshots')
parser.add_argument('--log_dir', default='/home/f517/PycharmProjects/ISPRS-pspnet-pytorch/log')
parser.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')

parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs to run')
parser.add_argument('--backend', type=str, default='resnet101', help='Feature extractor')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_type', type=str, default='poly', choices=['cosine', 'multistage', 'poly'])
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight_decay', default=1e-4)
parser.add_argument('--num_classes', type=int, help='Class number')
parser.add_argument('--crop_x', type=int, default=256, help='Horizontal random crop size')
parser.add_argument('--crop_y', type=int, default=256, help='Vertical random crop size')
parser.add_argument('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
parser.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
args = parser.parse_args()

models = {
    'squeezenet': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()

    # load net into Multi-GPU
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    ''''{}_{}_PSPNet_best_model.pth'.format(args.data, args.backend'''
    if snapshot is not None:
        print("Initializing weights from: {}...".format(snapshot))
        data, backend, _,_,_ = os.path.basename(snapshot).split('_')
        # epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        # logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    else:
        print("Training PSPNet from scratch...")
    net = net.cuda()
    return net, epoch


def train(data, save_path, snapshot, backend, crop_x, crop_y, batch_size, alpha, epochs, lr, milestones):

    save_path = os.path.abspath(os.path.expanduser(save_path))
    os.makedirs(save_path, exist_ok=True)
    
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''

    # training setting
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    composed_transforms_tr = transforms.Compose([tr.RandomSized(512),     #####
                                                 tr.RandomRotate(15),
                                                 tr.RandomHorizontalFlip(),
                                                 tr.Normalize(mean=mean, std=std),
                                                 tr.ToTensor()]
                                                )
    # testing setting
    composed_transforms_ts = transforms.Compose([tr.FixedResize(size=(512, 512)),
                                                 tr.Normalize(mean=mean, std=std),
                                                 tr.ToTensor()]
                                                )

    if data == 'ISPRS':
        print("Using ISPRS dataset......")
        CLASSES = ['impervious_surfaces', 'building', 'low_vegetation', 'tree', 'car', 'background']
        args.num_classes = len(CLASSES)
        ISPRS_train = ISPRS.ISPRSSegmentation(split='train', transform=composed_transforms_tr)
        ISPRS_val = ISPRS.ISPRSSegmentation(split='val', transform=composed_transforms_ts)
        db_train = ISPRS_train
        db_val = ISPRS_val

        # Setup confusion Metrics --> compting F1 score
        running_metrics_tr = metrics.runningScore(args.num_classes)
        running_metrics = metrics.runningScore(args.num_classes)
    elif data == "pascal_voc":
        print("Using pascal voc dataset......")
        args.num_classes = 22
        db_train = None
        db_val = None
        running_metrics_tr = metrics.runningScore(args.num_classes)
        running_metrics = metrics.runningScore(args.num_classes)
        pass
    elif data =="coco":
        args.num_classes = 10
        print("Using coco dataset......")
        db_train = None
        db_val = None
        running_metrics_tr = metrics.runningScore(args.num_classes)
        running_metrics = metrics.runningScore(args.num_classes)
        pass
    else:
        raise NotImplementedError
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # maybe need to adjust for the testloader batch_size
    testloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    print('the length of trainloader is: ', num_img_tr)
    print('the length of testloader is: ', num_img_ts)

    # build the model
    net, starting_epoch = build_network(snapshot, backend)

    # set the class-weight in Loss function
    class_weights = None

    # set training optimizer
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # seg_criterion = nn.NLLLoss2d(weight=class_weights)
    # cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    criterion = utils.cross_entropy2d

    running_loss_tr = 0.0
    global_step = 0
    best_iou = -100

    print("start training...")
    for epoch in range(starting_epoch, starting_epoch + epochs):
        start_time = timeit.default_timer()

        lr_ = utils.lr_poly(args.lr, epoch, args.epochs, 0.9)
        writer.add_scalar("learning_rate", scalar_value=lr_, global_step=epoch)
        optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=args.momentum, weight_decay=args.weight_decay)

        net.train()
        for ii, sample_batched in enumerate(trainloader):
            inputs, labels = sample_batched['image'], sample_batched['label']
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)

            inputs, labels = inputs.cuda(), labels.cuda()

            global_step += inputs.data.shape[0]  # shape[0] -->batch channel
            optimizer.zero_grad()
            outputs = net(inputs)
            pred_tr = outputs.data.max(1)[1].cpu().numpy()
            gt_tr = labels.data.cpu().numpy()
            running_metrics_tr.update(gt_tr, pred_tr)

            loss = criterion(outputs, labels.type(torch.cuda.LongTensor))
            running_loss_tr += loss.item()

            if ii % num_img_tr == (num_img_tr - 1):
                stop_time = timeit.default_timer()
                print("Epoch: [%d/%d] | Learning rate: %.6f | Excution time: %s | Loss: %.4f" %
                      (epoch + 1, args.epochs,
                       lr_,
                       str(stop_time - start_time),
                       running_loss_tr))
                writer.add_scalar("training loss", scalar_value=running_loss_tr, global_step=epoch)
                running_loss_tr = 0.0

            loss.backward()
            optimizer.step()
        score_tr, class_iou_tr, f1_scores_tr = running_metrics_tr.get_scores()
        print("Training: \n m-IOU is: {} \n F1 score is： {}".format(class_iou_tr, f1_scores_tr))

        net.eval()
        for i_val, val_batched in enumerate(testloader):
            with torch.no_grad():
                inputs_val = Variable(val_batched['image'].cuda())
                labels_val = Variable(val_batched['label'].cuda())

                outputs = net(inputs_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics.update(gt, pred)

        score, class_iou, f1_scores = running_metrics.get_scores()
        writer.add_scalar("Mean IoU", scalar_value=score['Mean IoU : \t'], global_step=epoch)
        writer.add_scalar("impervious_surfaces F1 score", scalar_value=f1_scores[0], global_step=epoch)
        writer.add_scalar("building F1 score", scalar_value=f1_scores[1], global_step=epoch)
        writer.add_scalar("low_vegetation F1 score", scalar_value=f1_scores[2], global_step=epoch)
        writer.add_scalar("tree F1 score", scalar_value=f1_scores[3], global_step=epoch)
        writer.add_scalar("car F1 score", scalar_value=f1_scores[4], global_step=epoch)
        writer.add_scalar("background F1 score", scalar_value=f1_scores[5], global_step=epoch)

        print("Validation: \n m-IOU is: {} \n F1 score is： {}".format(class_iou, f1_scores))
        print("Mean IoU is: {:.4f}".format(score['Mean IoU : \t']))
        print("overall acc is: {:.4f}".format(score['Overall Acc: \t']))
        print("=====================================================================================")
        running_metrics.reset()

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch + 1,
                     'model_state': net.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     }
            # torch.save(state, "{}_{}_best_model.pth".format(args.backend, args.data))
            filename = '{}_{}_PSPNet_best_model.pth'.format(args.data, args.backend)
            torch.save(net.state_dict(), os.path.join(save_path, filename))


if __name__ == '__main__':
    writer = SummaryWriter(log_dir=args.log_dir)
    # net, starting_epoch = build_network(args.snapshot, args.backend)
    train(data=args.data,
          save_path=args.save_path,
          snapshot=args.snapshot,
          backend=args.backend,
          crop_x=args.crop_x,
          crop_y=args.crop_y,
          batch_size=args.batch_size,
          alpha=args.alpha,
          epochs=args.epochs,
          lr=args.lr,
          milestones=args.milestones,
          )
    writer.close()
