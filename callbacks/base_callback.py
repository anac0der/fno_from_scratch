class Callback(object):
    """
    Base callback class. Each abstract method is called in the trainer's
    training loop at the appropriate time. 

    Taken from https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/training/callbacks.py

    Callbacks are stateful, meaning they keep track of a state and 
        update it throughout the lifetime of a Trainer class.
        Storing the state as a dict enables the Callback to keep track of
        references to underlying parts of the Trainer's process, such as 
        models, cost functions and output encoders
    """
    def __init__(self):
        self.state_dict = {}
    
    def _update_state_dict(self, **kwargs):
        self.state_dict.update(kwargs)

    def on_init_start(self, **kwargs):
        pass

    def on_init_end(self, *args, **kwargs):
        pass

    def on_before_train(self, *args, **kwargs):
        pass

    def on_train_start(self, *args, **kwargs):
        pass

    def on_epoch_start(self, *args, **kwargs):
        pass
    
    def on_batch_start(self, *args, **kwargs):
        pass

    def on_load_to_device(self, *args, **kwargs):
        pass
    
    def on_before_forward(self, *args, **kwargs):
        pass

    def on_before_loss(self, *args, **kwargs):
        pass
    
    def compute_training_loss(self, *args, **kwargs):
        raise NotImplementedError
    
    def on_batch_end(self, *args, **kwargs):
        pass
    
    def on_epoch_end(self, *args, **kwargs):
        pass
    
    def on_train_end(self, *args, **kwargs):
        pass

    def on_before_val(self, *args, **kwargs):
        pass

    def on_val_epoch_start(self, *args, **kwargs):
        pass
    
    def on_val_batch_start(self, *args, **kwargs):
        pass

    def on_before_val_loss(self, **kwargs):
        pass
    
    def compute_val_loss(self, *args, **kwargs):
        pass
    
    def on_val_batch_end(self, *args, **kwargs):
        pass

    def on_val_epoch_end(self, *args, **kwargs):
        pass
    
    def on_val_end(self, *args, **kwargs):
        pass