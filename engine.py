"""
Train and eval functions used in main.py
"""
# Imports
#basics
import os
import itertools

#torch
import torch
from torchvision.utils import make_grid

#processing
import numpy as np
from tqdm.auto import tqdm

#tensorboard
from torch.utils.tensorboard import SummaryWriter

# Functions
#metric
@torch.no_grad()
def accuracy(logits, targets):
    """
    Compute the accuracy for given logits and targets.

    Parameters
    ----------
    logits : (N, K) torch.Tensor
        A mini-batch of logit vectors from the network.
    targets : (N, ) torch.Tensor
        A mini_batch of target scalars representing the labels.

    Returns
    -------
    acc : () torch.Tensor
        The accuracy over the mini-batch of samples.
    """
    return (logits.argmax(dim=-1) == targets).float().mean()

#iterator functions
def _forward(network, data_loader, criterion):
    device = next(network.parameters()).device

    for batch, y_true in data_loader:
        batch, y_true = batch.to(device), y_true.to(device)
        logits = network(batch)
        loss = criterion(logits, y_true).mean()

        del y_true
        yield loss

@torch.no_grad()
def _forward_eval(network, data_loader, criterion, acc_criterion=accuracy):
    device = next(network.parameters()).device

    def unbatched_predictions():
        for batch, y_true, ids in data_loader:
            batch, y_true = batch.to(device), y_true.to(device)
            logits = network(batch)
            losses = criterion(logits, y_true.to(device))
            yield from zip(ids, logits, losses, y_true)

    def aggregated_by_id():
        for id, predictions in itertools.groupby(unbatched_predictions(),
                                                 key=lambda x: x[0]):
            ids, logits, losses, y_true = zip(*predictions)
            yield (id, sum(logits) / len(logits),
                   sum(losses) / len(losses), y_true[0])

    for id, logits, loss, y_true in aggregated_by_id():
        acc = acc_criterion(logits, y_true)
        yield loss, acc

#main training/eval functions
@torch.no_grad()
def evaluate(network, data_loader, criterion, tqdm_batch=None):
    losses = []
    accs = []

    network.eval()
    if tqdm_batch is not None: tqdm_batch.reset()
    for loss, acc in _forward_eval(network, data_loader, criterion):
        losses.append(float(loss.item()))
        accs.append(float(acc.item()))
        if tqdm_batch is not None: tqdm_batch.update()
    if tqdm_batch is not None: tqdm_batch.refresh()

    torch.cuda.empty_cache()
    return losses, accs


@torch.enable_grad()
def update(network, data_loader, criterion, optimizer, warmup_sched=None,
           tqdm_batch=None):
    a_loss = []
    nans_in_a_row = 0

    network.train()
    if tqdm_batch is not None: tqdm_batch.reset()
    for loss in _forward(network, data_loader, criterion):
        if not np.isfinite(loss.item()):
            nans_in_a_row += 1
            if nans_in_a_row < 5:
                continue
            else:
                raise RuntimeError("Training error is NaN! Stopping training.")
        else:
            nans_in_a_row = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if warmup_sched is not None:
            warmup_sched.step()

        a_loss.append(float(loss.detach().item()))
        if tqdm_batch is not None: tqdm_batch.update()
    if tqdm_batch is not None: tqdm_batch.refresh()

    #torch.cuda.empty_cache()
    return a_loss

#spectrograms for tb
@torch.no_grad()
def compute_frontend_example(network, data_loader):
    network.eval()
    device = next(network.parameters()).device
    batch, *_ = next(iter(data_loader))
    output = network._frontend(batch.to(device))
    if output.ndim == 3:
        output = output.unsqueeze(1)  # add singleton channel dimension
    if output.shape[1] > 1:
        # stack channels as bands
        output = output.reshape(output.shape[0], 1, -1, output.shape[-1])
    return output

class WarmupLR(object):
    def __init__(self, optimizer, warmup_steps, steps_done=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        if not steps_done:
            for group in self.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        self.steps_done = steps_done - 1
        self.step()

    def step(self):
        self.steps_done += 1
        if self.steps_done <= self.warmup_steps:
            factor = (self.steps_done + 1) / (self.warmup_steps + 1)
            for group in self.optimizer.param_groups:
                group['lr'] = factor * group['initial_lr']

def train(network, loader_train, loader_val, loader_test, path,
          criterion, optimizer,
          num_epochs, starting_epoch= None,
          save_every = 3, overwrite_save = True,  #overwrite_save overwrites current model, but also saves best model (based on val_acc),
          save_best_model = 'acc',  #save best model based on acc or loss of validation set
          test_every_epoch=False,  # whether to compute test metrics after every epoch
          scheduler = None, scheduler_item = 'acc', scheduler_min_lr = 1e-5,  #scheduler_item can be loss or acc
          warmup_steps=0,
          tqdm_on = True,  #print epoch in console/notebook
          model_name= 'default', writer_add_frontendparams = True): # TB parameters
    
    #quick asserts
    assert save_best_model == 'loss' or save_best_model == 'acc', 'save_best_model must be "loss" or "acc" of the validation set'
    assert scheduler_item == 'loss' or scheduler_item == 'acc' or scheduler_item == None, 'scheduler_item must be "loss" or "acc" of the validation set or None if no scheduler is used'
    
    #set starting epoch
    model_path = os.path.join(path, 'models', model_name)
    if not os.path.isdir(os.path.join(path, 'models')): os.mkdir(os.path.join(path, 'models'))
    if not os.path.isdir(model_path): os.mkdir(model_path)
    start = 1 if starting_epoch == None else starting_epoch

    #set warmup schedule
    if warmup_steps:
        warmup_sched = WarmupLR(optimizer, warmup_steps=warmup_steps,
                                steps_done=(start - 1) * len(loader_train))
    else:
        warmup_sched = None

    #tqdm
    if tqdm_on:
        tqdm_epoch = tqdm(range(start, num_epochs + 1), desc= 'Epochs')
        tqdm_train = tqdm(total=len(loader_train), desc= 'Train_Batch', leave= False)
        tqdm_val = tqdm(total=len(getattr(loader_val.dataset, 'dataset', loader_val.dataset)), desc= 'Val_Batch', leave= False)
        if test_every_epoch:
            tqdm_test = tqdm(total=len(getattr(loader_test.dataset, 'dataset', loader_test.dataset)), desc= 'Test_Batch', leave= False)
    else:
        tqdm_train = tqdm_val = tqdm_test = None

    #setup Tensorboard writer
    writer_path = os.path.join(path, 'runs', model_name)
    writer = SummaryWriter(log_dir=writer_path)

    #add start weights to the writer
    if start == 1:
        if writer_add_frontendparams and getattr(network, '_frontend', None):
            for name, weight in network._frontend.named_parameters():
                try:
                    writer.add_histogram(name, weight, 0)
                except ValueError:
                    pass

    #get basline val acc
    best_loss_val = None
    best_acc_val = None
    
    #training loop
    for epoch in range(start, num_epochs + 1):
        #log frontend output in Tensorboard
        if getattr(network, '_frontend', None):
            writer.add_image(
                "frontend",
                make_grid(compute_frontend_example(network,
                                                   loader_val)[:16].cpu(),
                          normalize=True),
                epoch)

        #train, test, evaluate
        if tqdm_on:
            tqdm_epoch.set_description(f"Epoch {epoch}")
        loss_train = update(network, loader_train, criterion, optimizer, warmup_sched, tqdm_train)
        loss_val, acc_val = evaluate(network, loader_val, criterion, tqdm_val)
        if test_every_epoch:
            loss_test, acc_test = evaluate(network, loader_test, criterion, tqdm_test)

        #Calc mean metrics
        mean_loss = np.mean(loss_train).item()
        mean_loss_val = np.mean(loss_val).item()
        mean_acc_val = np.mean(acc_val).item()
        if test_every_epoch:
            mean_loss_test = np.mean(loss_test).item()
            mean_acc_test = np.mean(acc_test).item()
        #set best if not already done
        best_loss_val = mean_loss_val if best_loss_val is None else best_loss_val
        best_acc_val = mean_acc_val if best_acc_val is None else best_acc_val
        
        #Tensorboard writing
        writer.add_scalar("loss/train loss", mean_loss, epoch)
        writer.add_scalar("loss/val loss", mean_loss_val, epoch)
        writer.add_scalar("acc/val acc", mean_acc_val, epoch)
        if test_every_epoch:
            writer.add_scalar("loss/test loss", mean_loss_test, epoch)
            writer.add_scalar("acc/test acc", mean_acc_test, epoch)
            writer.add_scalar("misc/global lr", optimizer.param_groups[0]["lr"], epoch)

        #add weights and gradient movement to the writer
        if writer_add_frontendparams and getattr(network, '_frontend', None):
            for name, weight in network._frontend.named_parameters():
                try:
                    writer.add_histogram(name, weight, epoch)
                    writer.add_histogram(f'{name}.grad',weight.grad, epoch)
                except ValueError:
                    pass

        #update tqdm or print to console
        if tqdm_on:
            test_metrics = (dict(loss_test=mean_loss_test, acc_test=100. * mean_acc_test)
                            if test_every_epoch else {})
            tqdm_epoch.set_postfix(loss=mean_loss, 
                                   loss_val=mean_loss_val,
                                   acc_val=100. * mean_acc_val,
                                   **test_metrics)
        else:
            test_metrics = (("Test_Loss: {:.5f}".format(mean_loss_test),
                             "Test_Acc: {:.5f}".format(mean_acc_test))
                            if test_every_epoch else ())
            print( "Epoch: {}/{} - ".format(epoch, num_epochs), #print out status
                   "Avg_Loss: {:.5f} -".format(mean_loss),
                   "Eval_Loss: {:.5f}".format(mean_loss_val),
                   "Eval_Acc: {:.5f}".format(mean_acc_val),
                   *test_metrics)
        
        
        #save network every now and then
        if epoch % save_every == 0:
            path_network_save = os.path.join(model_path, "net_checkpoint.pth") if overwrite_save else os.path.join(model_path, "net_checkpoint_e" + str(epoch) + ".pth")
            torch.save({'epoch': epoch, 
                        'network': network.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'scheduler':  scheduler.state_dict() if scheduler is not None else None},
                       path_network_save)

        #save if best model
        if save_best_model == 'acc':
            if mean_acc_val > best_acc_val:
                torch.save({'epoch': epoch, 
                            'network': network.state_dict(), 
                            'optimizer': optimizer.state_dict(),
                            'scheduler':  scheduler.state_dict() if scheduler is not None else None},
                           os.path.join(model_path, "net_best_model.pth"))
                best_acc_val = mean_acc_val
        if save_best_model == 'loss':
            if mean_loss_val < best_loss_val:
                torch.save({'epoch': epoch, 
                            'network': network.state_dict(), 
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict() if scheduler is not None else None},
                           os.path.join(model_path, "net_best_model.pth"))
                best_loss_val = mean_loss_val
                
        #lr scheduler
        if scheduler is not None:
            if scheduler_item == 'acc':
                scheduler.step(mean_acc_val)
            if scheduler_item == 'loss':
                scheduler.step(mean_loss_val)
            if optimizer.param_groups[0]['lr'] < scheduler_min_lr:
                print('Learning rate fell below threshold. Stopping training.')
                break
        
        #update tqdm bar
        if tqdm_on: tqdm_epoch.update()


    #when done training
    torch.save({'epoch': epoch, 
                'network': network.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None},
               os.path.join(model_path, "net_last_model.pth"))

    if not test_every_epoch:
        if tqdm_on:
            tqdm_test = tqdm(total=len(getattr(loader_test.dataset, 'dataset', loader_test.dataset)), desc= 'Test_Batch', leave= False)
        loss_test, acc_test = evaluate(network, loader_test, criterion, tqdm_test)
        mean_loss_test = np.mean(loss_test).item()
        mean_acc_test = np.mean(acc_test).item()
        writer.add_scalar("loss/test loss", mean_loss_test, epoch)
        writer.add_scalar("acc/test acc", mean_acc_test, epoch)
        writer.add_scalar("misc/global lr", optimizer.param_groups[0]["lr"], epoch)
        print("Test_Loss: {:.5f}".format(mean_loss_test),
              "Test_Acc: {:.5f}".format(mean_acc_test))

    # ensure tensorboard has written out everything
    # (https://github.com/pytorch/pytorch/issues/24234)
    writer.flush()

    #save final metrics
    with open(os.path.join(model_path, "final_metrics.txt"), 'w') as f:
        f.writelines("%s=%g\n" % item for item in [
            ('eval_loss', mean_loss_val),
            ('eval_acc', mean_acc_val),
            ('test_loss', mean_loss_test),
            ('test_acc', mean_acc_test)])
