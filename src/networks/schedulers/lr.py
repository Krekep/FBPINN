from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LRScheduler


class WarmupReduceLROnPlateau(LRScheduler):
    """
    Scheduler that first performs a warmup (linear increase in LR),
    and after the warmup completes, it operates as ReduceLROnPlateau.
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8,
                 warmup_epochs=0, warmup_start_factor=0.1):
        """
        Parameters
        ----------
            optimizer
                PyTorch optimizer
            mode: str
                mode for torch.ReduceLROnPlateau
            factor: float
                factor for torch.ReduceLROnPlateau
            patience: int
                patience for torch.ReduceLROnPlateau
            threshold: float
                threshold for torch.ReduceLROnPlateau
            threshold_mode: str
                threshold_mode for torch.ReduceLROnPlateau
            cooldown: int
                cooldown for torch.ReduceLROnPlateau
            min_lr: float|list
                min_lr for torch.ReduceLROnPlateau
            eps: float
                eps for torch.ReduceLROnPlateau
            warmup_epochs: int
                number of warmup epochs
            warmup_start_factor: float
                multiplier for obtaining the starting LR in the scheduler (start_lr = LR * base_lr)
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.epoch = 0
        self._reduce_scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            threshold=threshold, threshold_mode=threshold_mode,
            cooldown=cooldown, min_lr=min_lr, eps=eps
        )
        super(WarmupReduceLROnPlateau, self).__init__(optimizer)

    def step(self, metrics=None):
        """
        Called at the end of each epoch.
        Update epoch and then update LR
        If metrics is None, only the `epoch` is incremented.
        Otherwise, the epoch is updated first, and then the metric is processed.
        """
        self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            progress = self.epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                target_lr = base_lr
                start_lr = base_lr * self.warmup_start_factor
                param_group['lr'] = start_lr + (target_lr - start_lr) * progress
        else:
            if metrics is not None:
                self._reduce_scheduler.step(metrics)

    def state_dict(self):
        state = {
            'epoch': self.epoch,
            'base_lrs': self.base_lrs,
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_factor': self.warmup_start_factor,
            'reduce_scheduler': self._reduce_scheduler.state_dict()
        }
        return state

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.base_lrs = state_dict['base_lrs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_start_factor = state_dict['warmup_start_factor']
        self._reduce_scheduler.load_state_dict(state_dict['reduce_scheduler'])
