import os
import logging
import argparse
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
from tqdm import tqdm
from pspnet import PSPNet
import utils
import metrics

parser = argparse.ArgumentParser(description="Pytorch PSPNet parameters")
parser.add_argument('--data', type=str, default="ISPRS",
                    help='Path to dataset folder')
parser.add_argument('--models-path', type=str, default="/home/liujiahui/PycharmProjects/pspnet-pytorch/model",
                    help='Path for storing model snapshots')
parser.add_argument('--backend', type=str, default='resnet18', help='Feature extractor')
parser.add_argument('--num_classes', type=int, help='Class number')
parser.add_argument('--snapshot', type=str, default=None, help='Path to pretrained weights')
parser.add_argument('--crop_x', type=int, default=256, help='Horizontal random crop size')
parser.add_argument('--crop_y', type=int, default=256, help='Vertical random crop size')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs to run')
parser.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
parser.add_argument('--start-lr', type=float, default=0.001)
parser.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
args = parser.parse_args()

models = {
    'squeezenet': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(args.num_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(args.num_classes,sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(args.num_classes,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(args.num_classes,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(args.num_classes,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    # load net into GPU
    # net = nn.DataParallel(net)
    net.cuda()

    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def train(data, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones):

    #data_path = os.path.abspath(os.path.expanduser(data_path))
    models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
    
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    # training setting
    composed_transforms_tr = transforms.Compose([tr.RandomSized(512),     #####
                                                 tr.RandomRotate(15),
                                                 tr.RandomHorizontalFlip(),
                                                 tr.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225)),
                                                 tr.ToTensor()]
                                                )
    # testing setting
    composed_transforms_ts = transforms.Compose([tr.FixedResize(size=(512, 512)),
                                                 tr.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225)),
                                                 tr.ToTensor()]
                                                )

    if data == 'ISPRS':
        print("Using ISPRS dataset......")
        args.num_classes = 6
        ISPRS_train = ISPRS.ISPRSSegmentation(split='train', transform=composed_transforms_tr)
        ISPRS_val = ISPRS.ISPRSSegmentation(split='val', transform=composed_transforms_ts)
        db_train = ISPRS_train
        db_val = ISPRS_val

        # Setup confusion Metrics --> compting F1 score
        running_metrics = metrics.runningScore(args.num_classes)
    elif data == "pascal_voc":
        print("Using pascal voc dataset......")
        args.num_classes = 22
        db_train = None
        db_val = None
        running_metrics = metrics.runningScore(args.num_classes)
        pass
    elif data =="coco":
        args.num_classes = 10
        print("Using coco dataset......")
        db_train = None
        db_val = None
        running_metrics = metrics.runningScore(args.num_classes)
        pass
    else:
        raise NotImplementedError

    net, starting_epoch = build_network(snapshot, backend)
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # maybe need to adjust for the testloader batch_size
    testloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    print('the length of trainloader is: ', num_img_tr)
    print('the length of testloader is: ', num_img_ts)

    # train_loader, class_weights, n_images = None, None, None

    # set the class-weight in Loss function
    class_weights = None
    # set training optimizer
    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])

    # seg_criterion = nn.NLLLoss2d(weight=class_weights)
    # cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    criterion = utils.cross_entropy2d

    global_step = 0

    best_iou = -100
    for epoch in range(starting_epoch, starting_epoch + epochs):
        epoch_losses = []
        # train_iterator = tqdm(loader, total=max_steps // batch_size + 1)
        # set the model in training mode
        net.train()
        for ii, sample_batched in enumerate(trainloader):
            inputs, labels = sample_batched['image'], sample_batched['label']
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            # print('the input image shape is: ', inputs.shape)
            # print('the input label shape is: ', labels.shape)

            inputs, labels = inputs.cuda(), labels.cuda()

            global_step += inputs.data.shape[0]  # shape[0] -->batch channel
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels.type(torch.cuda.LongTensor))
            # two-loss training stradgy
            # seg_loss, cls_loss = seg_criterion(out, labels), cls_criterion(out_cls, labels)
            # loss = seg_loss + alpha * cls_loss
            epoch_losses.append(loss.data[0])
            # if (ii + 1) % 20 ==0:
            #     status = '{0} loss = {1:0.5f} avg = {2:0.5f}, lr = {5:0.7f}'.format(
            #         epoch + 1, loss.data[0], np.mean(epoch_losses), scheduler.get_lr()[0])
            #     print(status)
            if (ii + 1) % 1 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args.epochs, loss.data[0]))
            # train_iterator.set_description(status)
            loss.backward()
            optimizer.step()

        net.eval()
        for i_val, val_batched in enumerate(testloader):
            inputs_val = Variable(val_batched['image'].cuda(), volatile=True)
            labels_val = Variable(val_batched['label'].cuda(), volatile=True)

            outputs = net(inputs_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        print("epoch: {}, m-IOU is: {}".format(epoch, class_iou))
        running_metrics.reset()

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch + 1,
                     'model_state': net.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     }
            torch.save(state, "{}_{}_best_model.pkl".format(args.backend, args.data))
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1)])))
        scheduler.step()

        # train_loss = np.mean(epoch_losses)


if __name__ == '__main__':

    train(data=args.data,
          models_path=args.models_path,
          backend=args.backend,
          snapshot=args.snapshot,
          crop_x=args.crop_x,
          crop_y=args.crop_y,
          batch_size=args.batch_size,
          alpha=args.alpha,
          epochs=args.epochs,
          start_lr=args.start_lr,
          milestones=args.milestones,
          )
