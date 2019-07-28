import math
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def adjust_learning_rate(args, optimizer, iter_, max_iter, power=0.9,
                         batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = max_iter * nBatch
        T_cur = (iter_ % max_iter) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if iter_ >= max_iter * 0.75:
                lr *= decay_rate**2
            elif iter_ >= max_iter * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (iter_// 30))
    elif method == 'poly':
        lr = args.lr * ((1 - float(iter_) / max_iter) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr