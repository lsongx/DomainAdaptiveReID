from __future__ import print_function, absolute_import
import argparse
import time
import os.path as osp
import os 
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.serialization import load_checkpoint, save_checkpoint

from sklearn.cluster import DBSCAN
from reid.rerank import re_ranking


def get_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training and validation images in target dataset
    train_set = dataset.trainval 
    num_classes = dataset.num_trainval_ids

    transformer = T.Compose([
        T.Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, extfeat_loader, test_loader


def get_source_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training images on source dataset
    train_set = dataset.train
    num_classes = dataset.num_train_ids

    transformer = T.Compose([
        T.Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, extfeat_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    # get source data
    src_dataset, src_extfeat_loader = \
        get_source_data(args.src_dataset, args.data_dir, args.height,
                        args.width, args.batch_size, args.workers)
    # get target data
    tgt_dataset, num_classes, tgt_extfeat_loader, test_loader = \
        get_data(args.tgt_dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    # Hacking here to let the classifier be the number of source ids
    if args.src_dataset == 'dukemtmc':
        model = models.create(args.arch, num_classes=632, pretrained=False)
    elif args.src_dataset == 'market1501':
        model = models.create(args.arch, num_classes=676, pretrained=False)
    else:
        raise RuntimeError('Please specify the number of classes (ids) of the network.')

    # Load from checkpoint
    if args.resume:
        print('Resuming checkpoints from finetuned model on another dataset...\n')
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise RuntimeWarning('Not using a pre-trained model.')
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model, print_freq=args.print_freq)
    print("Test with the original model trained on source domain:")
    evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    if args.evaluate:
        return

    # Criterion
    criterion = [
        TripletLoss(args.margin, args.num_instances).cuda(),
        TripletLoss(args.margin, args.num_instances).cuda(),
    ]

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, 
    )

    # training stage transformer on input images
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((args.height,args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])

    # Start training
    for iter_n in range(args.iteration):
        if args.lambda_value == 0:
            source_features = 0
        else:
            # get source datas' feature
            source_features, _ = extract_features(model, src_extfeat_loader, print_freq=args.print_freq)
            # synchronization feature order with src_dataset.train
            source_features = torch.cat([source_features[f].unsqueeze(0) for f, _, _ in src_dataset.train], 0) 

        # extract training images' features
        print('Iteration {}: Extracting Target Dataset Features...'.format(iter_n+1))
        target_features, _ = extract_features(model, tgt_extfeat_loader, print_freq=args.print_freq)
        # synchronization feature order with dataset.train
        target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in tgt_dataset.trainval], 0) 
        # calculate distance and rerank result
        print('Calculating feature distances...') 
        target_features = target_features.numpy()
        rerank_dist = re_ranking(
            source_features, target_features, lambda_value=args.lambda_value) 
        if iter_n==0:
            # DBSCAN cluster
            tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            top_num = np.round(args.rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps,min_samples=4,metric='precomputed', n_jobs=8)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - 1 
        print('Iteration {} have {} training ids'.format(iter_n+1, num_ids))
        # generate new dataset
        new_dataset = []
        for (fname, _, _), label in zip(tgt_dataset.trainval, labels):
            if label==-1:
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname,label,0)) 
        print('Iteration {} have {} training images'.format(iter_n+1, len(new_dataset)))

        train_loader = DataLoader(
            Preprocessor(new_dataset, root=tgt_dataset.images_dir,
                         transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(new_dataset, args.num_instances),
            pin_memory=True, drop_last=True)

        # train model with new generated dataset
        trainer = Trainer(model, criterion, print_freq=args.print_freq)
        evaluator = Evaluator(model, print_freq=args.print_freq)
        # Start training
        for epoch in range(args.epochs):
            trainer.train(epoch, train_loader, optimizer)
        # Evaluate
        
        if iter_n == 19:
            args.rho *= 0.7
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
            criterion = [
                TripletLoss(args.margin, args.num_instances, False).cuda(),
                TripletLoss(args.margin, args.num_instances, False).cuda(),
            ]
            mean_ap, rank1, rank5, rank10 = evaluator.evaluate(
                test_loader, tgt_dataset.query, tgt_dataset.gallery)

    mean_ap, rank1, rank5, rank10 = evaluator.evaluate(
        test_loader, tgt_dataset.query, tgt_dataset.gallery)
    return (mean_ap, rank1, rank5, rank10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('--src_dataset', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('--tgt_dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=models.names())
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help="balancing parameter, default: 0.1")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    # optimizer
    parser.add_argument('--lr', type=float, default=6e-5,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default = '')
    parser.add_argument('--evaluate', type=int, default=0,
                        help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=70)
    # metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='')

    args = parser.parse_args()
    mean_ap, rank1, rank5, rank10 = main(args)
    results_file = np.asarray([mean_ap, rank1, rank5, rank10])
    file_name = time.strftime("%H%M%S", time.localtime())
    file_name = osp.join(args.logs_dir, file_name)
    np.save(file_name, results_file)
