import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .metrics import MulticlassMetrics

class BaselineCallback():
    """
    Base class for a generic design of callbacks.
    """
    def __init__(self):
      pass

    def on_train_batch_begin(self, batch, logs={}):
      pass

    def on_test_batch_begin(self, batch, logs={}):
      pass

    def on_train_batch_end(self, batch, logs={}):
      pass

    def on_test_batch_end(self, batch, logs={}):
      pass

    def on_epoch_begin(self, epoch, logs={}):
      pass

    def on_epoch_end(self, epoch, logs={}):
      pass

    def on_train_end(self, logs={}):
      pass

    def on_test_end(self, logs={}):
      pass

class ModelSaveCallback(BaselineCallback):
    def __init__(self, period, path):
        """
        Callback to save the model at regular intervals.
        :param period: Save the model every `period` epochs.
        :param path: Path to save the model.
        """
        self.period = period
        self.path = path
        self.history = {"ModelSaveCallback":{}}

    def on_epoch_end(self, epoch, logs):
        """
        Function to be called at the end of each epoch.
        :param epoch: Current epoch number.
        :param model: The model to be saved.
        """
        if (epoch + 1) % self.period == 0:
            torch.save(logs['model'].state_dict(), self.path)
            print(f"Model saved at epoch {epoch + 1}")
            self.history["ModelSaveCallback"][epoch] = f"Model saved at epoch {epoch + 1}"

class MetricsCallback(BaselineCallback):
    def __init__(self,log_dir = "/content/logs", base_epoch=0, skipped_metrics_print_interval=5, total_train_samples=1, total_val_samples=1, batch_size=1):
        self.t_steps_per_epoch = total_train_samples // batch_size + int((total_train_samples % batch_size) != 0)
        self.v_steps_per_epoch = total_val_samples // batch_size + int((total_val_samples % batch_size) != 0)
        self.reset_pbar()

        self.train_metrics_tracker = MulticlassMetrics()
        self.val_metrics_tracker = MulticlassMetrics()
        self.history = {"MetricsCallback": {}}
        self.skipped_metrics_print_interval = skipped_metrics_print_interval

        # Initialize TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.base_epoch = base_epoch

    def reset_pbar(self):
        self.pbar = tqdm(
            total=self.t_steps_per_epoch,
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
        )

    def get_verbose_description(self, mode='train', epoch=None):
        verbose_text = ""
        metrics = self.train_metrics_tracker.result() if mode == 'train' else self.val_metrics_tracker.result()

        for k, v in metrics.items():
            # If skip is present in the key then we skip that attribute at the Tensorboard and Regular logs
            if "skip" in k:
                continue
            verbose_text += f"{mode}_{k}: {v:.4f} | "

            # Log to TensorBoard if epoch is provided
            if epoch is not None:
                self.writer.add_scalar(f"{mode}/{k}", v, self.base_epoch+epoch)

        return verbose_text

    def print_skipped_metrics(self, mode='train'):
        print(f"Skipped Metrics [{mode}]:")
        metrics = self.train_metrics_tracker.result() if mode == 'train' else self.val_metrics_tracker.result()
        # Print skipped metrics
        for k, v in metrics.items():
            if "skip" in k:
                print(f"{k}: {v:.4f}")

    def on_train_batch_end(self, batch, logs={}):
        self.train_metrics_tracker.update_state(**logs)
        verbose = self.get_verbose_description(mode='train')
        self.pbar.set_description(verbose)
        self.pbar.update()

    def on_test_batch_end(self, batch, logs={}):
        self.val_metrics_tracker.update_state(**logs)

    def on_epoch_begin(self, epoch, logs={}):
        self.reset_pbar()
        self.train_metrics_tracker.reset_state()
        self.val_metrics_tracker.reset_state()
        print(f"\n[START OF RESULT]\nEpoch {epoch+1}")
        self.history["MetricsCallback"][epoch] = {}

    def on_epoch_end(self, epoch, logs={}):
        train_verbose = self.get_verbose_description(mode='train', epoch=epoch)
        val_verbose = self.get_verbose_description(mode='val', epoch=epoch)
        print("\n" + train_verbose + "\n" + val_verbose + "\n")
        self.history["MetricsCallback"][epoch]["train"] = train_verbose
        self.history["MetricsCallback"][epoch]["val"] = val_verbose

        # Print skipped metrics
        if epoch % self.skipped_metrics_print_interval == 0:
            self.print_skipped_metrics(mode='train')
            self.print_skipped_metrics(mode='val')

        print("[END OF RESULT]")

        # Flush the writer to ensure logs are saved
        self.writer.flush()

    def __del__(self):
        self.writer.close()  # Close the TensorBoard writer when the object is deleted
