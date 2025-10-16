import torch
import os
from .metrics import calculate_f1_precision_recall_accuracy, calculate_class_wise_accuracy
from .visualization import run_layerwise_correlations_plot_steps

class Trainer:
  """
  Class to manage the training and validation process of a PyTorch model.
  """
  def __init__(self,
               device,
               model,
               callbacks,
               vis_save_path,
               num_labs) -> None:
      """
      Initializes the Trainer class.

      Args:
          device (str): The device to train the model on (e.g., 'cpu' or 'cuda').
          model (nn.Module): The PyTorch model to train.
          callbacks (list): A list of callback objects to be executed during training.
          vis_save_path (str): Path to save visualizations.
          num_labs (int): Number of classes.
      """
      self.device = device
      self.model = model
      self.callbacks = callbacks
      self.history = {}
      self.total_epochs = None
      self.vis_save_path = vis_save_path
      self.num_labs = num_labs

  def compile(self,
              loss_obj,
              optimizer_obj):
      """
      Compiles the Trainer by setting the loss function and optimizer.

      Args:
          loss_obj (nn.Module): Our custom loss class object.
          optimizer_obj (torch.optim.Optimizer): The optimizer object to use for training.
      """
      self.loss_obj = loss_obj
      self.optimizer_obj = optimizer_obj


  def train_one_epoch(self, epoch, train_loader):
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        """
        for b, batch in enumerate(train_loader):
            for callback in self.callbacks:
                callback.on_train_batch_begin(b, logs={})

            [x, y_true] = batch
            x, y_true = x.to(self.device), y_true.to(self.device)
            x = x.type(torch.float)

            if b==0:
                  os.makedirs(self.vis_save_path, exist_ok=True)
                  run_layerwise_correlations_plot_steps(
                      self.model,
                      x,
                      title=f"Epoch {epoch+1} Batch {b+1}",
                      vis_save_path=f"{self.vis_save_path}/epoch_{epoch+1}_batch_{b+1}.png",
                      json_save_path=f"{self.vis_save_path}/corr.json",
                      pltflg=False
                  )

            y_true_oh = torch.nn.functional.one_hot(y_true.type(torch.long), num_classes=self.num_labs)
            y_true_oh = y_true_oh.type(torch.float)

            self.model.train()
            self.optimizer_obj.zero_grad()
            y_pred = self.model(x)
            total_loss_value = self.loss_obj(y_true_oh, y_pred)
            total_loss_value.backward()
            self.optimizer_obj.step()
            self.optimizer_obj.zero_grad()

            f1_score, precision, recall, acc = calculate_f1_precision_recall_accuracy(y_true_oh, y_pred)
            class_wise_accs = calculate_class_wise_accuracy(y_true_oh, y_pred)

            logs = {
                'total_loss_value': total_loss_value.item(),
                'precision': precision.item(),
                'recall': recall.item(),
                'f1_score': f1_score.item(),
                'acc': acc.item(),
                **class_wise_accs
            }

            for callback in self.callbacks:
                callback.on_train_batch_end(b, logs=logs)

  def validate_one_epoch(self, epoch, val_loader):
        """
        Validates the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            val_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        """
        self.model.eval()
        with torch.no_grad():
            for b, batch in enumerate(val_loader):
                for callback in self.callbacks:
                    callback.on_test_batch_begin(b, logs={})

                [x, y_true] = batch
                x, y_true = x.to(self.device), y_true.to(self.device)
                x = x.type(torch.float)

                y_true_oh = torch.nn.functional.one_hot(y_true.type(torch.long), num_classes=self.num_labs)
                y_true_oh = y_true_oh.type(torch.float)

                y_pred = self.model(x)
                total_loss_value = self.loss_obj(y_true_oh, y_pred)
                f1_score, precision, recall, acc = calculate_f1_precision_recall_accuracy(y_true_oh, y_pred)
                classwise_accs = calculate_class_wise_accuracy(y_true_oh, y_pred)

                logs = {
                    'total_loss_value': total_loss_value.item(),
                    'precision': precision.item(),
                    'recall': recall.item(),
                    'f1_score': f1_score.item(),
                    'acc': acc.item(),
                    **classwise_accs
                }

                for callback in self.callbacks:
                    callback.on_test_batch_end(b, logs=logs)

  def fit(self, epochs, train_loader, val_loader):
        """
        Trains the model for a specified number of epochs and validates it after each epoch.

        Args:
            epochs (int): The total number of epochs to train for.
            train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
            val_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.

        Returns:
            dict: A dictionary containing the training history.
        """
        self.total_epochs = epochs
        for epoch in range(self.total_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, logs={})

            self.train_one_epoch(epoch, train_loader)
            self.validate_one_epoch(epoch, val_loader)

            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs={'model': self.model})
                self.history[callback.__class__.__name__] = callback.history

        return self.history
