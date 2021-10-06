# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmcv.utils import TORCH_VERSION, digit_version
from torch.utils import data
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook
import numpy as np

@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        super(TensorboardLoggerHook, self).before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        current_iter = self.get_iter(runner)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(self._modify_tag(tag), val, current_iter)
            else:
                self.writer.add_scalar(self._modify_tag(tag), val, current_iter)
        if 'train_visualize_output' in runner.outputs:
            img = runner.outputs['train_visualize_output']
            for i in range(len(img)):
                self.writer.add_images('topk_train_' + str(i+1), img[i], current_iter, dataformats='HWC')
            
        if 'val_visualize_output' in runner.outputs:
            img = runner.outputs['val_visualize_output']
            for i in range(len(img)):
                self.writer.add_images('topk_val_' + str(i+1), img[i], current_iter, dataformats='HWC')
        
    @master_only
    def after_run(self, runner):
        self.writer.close()

    def _modify_tag(self, tag):
        if 'val' in tag:
            if 'train' in tag:
                if 'AR' in tag:
                    return tag.replace('val', 'train_AR')
                elif 'AP' in tag:
                    return tag.replace('val', 'train_AP')
            else:
                if 'loss' in tag:
                    return tag.replace('val', 'val_loss')
                elif 'AP' in tag:
                    return tag.replace('val', 'val_AP')
                elif 'AR' in tag:
                    return tag.replace('val', 'val_AR')
                elif 'top' in tag:
                    return tag.replace('val', 'val_toploss')
        elif 'train' in tag:
            if 'loss' in tag:
                return tag.replace('train', 'train_loss')
            elif 'AP' in tag:
                return tag.replace('train', 'train_AP')
            elif 'AR' in tag:
                return tag.replace('train', 'train_AR')
            elif 'top' in tag:
                return tag.replace('train', 'train_toploss')
        else: 
            return tag