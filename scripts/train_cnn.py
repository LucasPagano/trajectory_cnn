import argparse
import gc
import logging
import os
import pathlib
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from cnn.model_cnn import TrajEstimator
from cnn.model_cnn_moving_threshold import TrajEstimatorThreshold
from sgan.data.loader import data_loader
from sgan.losses import displacement_error, final_displacement_error
from sgan.losses import l2_loss
from sgan.utils import get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--delim', default="tab")
parser.add_argument('--loader_num_workers', default=0, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_epochs', default=1, type=int)

# Model Options
parser.add_argument('--moving_threshold', default=0, type=int)
parser.add_argument('--embedding_dim', default=32, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--dropout', default=0, type=float)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int)

parser.add_argument('--g_learning_rate', default=1e-3, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1, type=float)

# Output
parser.add_argument('--output_dir', default="save")
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='eth_50epoch')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=0, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    logdir = "tensorboard/" + args.dataset_name + "/" + str(args.num_epochs) + "_epoch_" + str(args.g_learning_rate) + "_lr"
    if os.path.exists(logdir):
        for file in os.listdir(logdir):
            os.unlink(os.path.join(logdir, file))

    writer = SummaryWriter(logdir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')
    checkpoint_path = os.path.join(args.output_dir, '%s_with_model.pt' % args.checkpoint_name)
    pathlib.Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    long_dtype, float_dtype = get_dtypes(args)

    if args.moving_threshold:
        generator = TrajEstimatorThreshold(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            num_layers=args.num_layers,
            dropout=args.dropout)
    else:
        generator = TrajEstimator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            num_layers=args.num_layers,
            dropout=args.dropout)

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    generator.train()

    logger.info('Here is the generator:')
    logger.info(generator)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size
    args.num_iterations = int(iterations_per_epoch * args.num_epochs)
    # log 100 points
    log_tensorboard_every = int(args.num_iterations * 0.01) - 1
    if log_tensorboard_every <= 0:
        #there are less than 100 iterations
        log_tensorboard_every = int(args.num_iterations) / 4

    logger.info(
        'There are {} iterations per epoch'.format(int(iterations_per_epoch))
    )

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'g_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'best_t_nl': None,
        }
    while t < args.num_iterations:
        gc.collect()

        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            losses_g = generator_step(args, batch, generator,
                                      optimizer_g, epoch)
            checkpoint['norm_g'].append(
                get_total_norm(generator.parameters())
            )

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save values for tensorboard
            if t % log_tensorboard_every == 0:
                for k, v in sorted(losses_g.items()):
                    writer.add_scalar(k, v, t)

                metrics_val = check_accuracy(
                    args, val_loader, generator, epoch
                )
                metrics_train = check_accuracy(
                    args, train_loader, generator, epoch, limit=True
                )
                to_keep = ["g_l2_loss_rel", "ade", "fde"]
                for k, v in sorted(metrics_val.items()):
                    if k in to_keep:
                        writer.add_scalar("val_" + k, v, t)

                for k, v in sorted(metrics_train.items()):
                    if k in to_keep:
                        writer.add_scalar("train_" + k, v, t)

            # Maybe save a checkpoint
            if t % args.checkpoint_every == 0 and t > 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, epoch
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, epoch, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')
                
            t += 1
            if t >= args.num_iterations:
                if args.moving_threshold:
                    logger.info("Non-moving trajectories : {}%, threshold : {}".format(round((float(generator.total_trajs_under_threshold) / generator.total_trajs) * 100), generator.threshold))
                break


def generator_step(args, batch, generator, optimizer_g, epoch):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, epoch)

    if args.l2_loss_weight > 0:
        g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
            pred_traj_fake_rel,
            pred_traj_gt_rel,
            loss_mask,
            mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:

        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    loss.backward()
    optimizer_g.step()
    optimizer_g.zero_grad()

    return losses


def check_accuracy(args, loader, generator, epoch, limit=False):
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, epoch)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    # metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
                total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
        pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
        loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
        pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
