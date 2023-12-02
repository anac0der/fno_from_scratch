from .base_callback import Callback
from torch.utils.tensorboard.writer import SummaryWriter

class TensorBoardCallback(Callback):
    """
    This callback writes train error and test losses to tensorboard logs.
    """
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.curr_val_epoch = 0
    
    def on_before_val(self, **kwargs):
        train_err = kwargs['train_err']
        epoch = kwargs['epoch']
        self.curr_val_epoch = epoch
        self.writer.add_scalar('train_err', train_err, epoch)
    
    def on_val_epoch_end(self, **kwargs):
        errors = kwargs['errors']
        for key in errors.keys():
             self.writer.add_scalar(key, errors[key], self.curr_val_epoch)