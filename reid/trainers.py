from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterions, print_freq=1):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterions = criterions
        self.print_freq = print_freq

    def train(self, epoch, data_loader, optimizer):
        self.model.train()
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = False

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets, epoch)
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            #add gradient clip for lstm
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs) #outputs=[x1,x2,x3]
        #new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)
        # x2 global feature cross entropy loss
        #loss_global = self.criterions[1](outputs[1], targets)
        loss_global, prec_global = self.criterions[1](outputs[1], targets, epoch)
        #prec_global, = accuracy(outputs[1].data, targets.data)
        #prec_global = prec_global[0]

        return loss_tri+loss_global, prec_global
