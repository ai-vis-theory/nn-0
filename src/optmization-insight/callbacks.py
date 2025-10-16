import tensorflow as tf
import numpy as np
import os
import shutil
import datetime
from tqdm import tqdm

from . import utils
from . import config
from .model import build_model

class OptimizationInsightCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_generator, **kwargs):
        super().__init__()
        self._train_generator = train_generator
        self.bin_count = kwargs.get("bin_count", config.BIN_COUNT)
        self.clip = kwargs.get("clip", config.CLIP)
        self.save_check_periodically = kwargs.get("save_check_periodically", config.SAVE_CHECK_PERIODICALLY)
        self.periodic_save_dir_path = kwargs.get("periodic_save_dir_path", config.PERIODIC_SAVE_DIR_PATH)
        self.period = kwargs.get("period", config.PERIOD)
        self._load_model_path = kwargs.get("load_model_path", config.LOAD_MODEL_PATH)
        self._load_interim_results_path = kwargs.get("load_interim_results_path", config.LOAD_INTERIM_RESULTS_PATH)

        self.interim_results = self._initialize_interim_results()
        self.summary_writer = {}
        self.load_model_and_results()

        _dummy_gradient, _dummy_sec_gradient = self._calculate_dummy_grads()

        self.variables = {
            "model": self.model,
            "g_cur": utils.flatten_grad(_dummy_gradient),
            "g_prev": utils.flatten_grad(_dummy_gradient),
            "g_cur_09avg": utils.flatten_grad(_dummy_gradient),
            "g_prev_09avg": utils.flatten_grad(_dummy_gradient),
            "w_cur": utils.flatten_grad(self.model.trainable_variables),
            "w_prev": utils.flatten_grad(self.model.trainable_variables),
            "w_cur_09avg": utils.flatten_grad(self.model.trainable_variables),
            "w_prev_09avg": utils.flatten_grad(self.model.trainable_variables),
        }

        self.t_steps_per_epoch = len(self._train_generator)
        self.reset_pbar()
        self.update_logs_directory()
        self.epoch = len(self.interim_results["epoch_wise"]['train_mean_loss'])

        if self.periodic_save_dir_path and not os.path.exists(self.periodic_save_dir_path):
            os.makedirs(self.periodic_save_dir_path)

    def _initialize_interim_results(self):
        batch_wise_keys = [
            "g_cur_hist", "train_loss", "no_zero_e7_grad1", "no_zero_e7_grad2",
            "no_zero_e7_grad_retained", "no_zero_e7_grad_released", "pos_zero_e7_grad2",
            "fraction_zeros_e7_released", "no_zero_e5_grad1", "no_zero_e5_grad2",
            "no_zero_e5_grad_retained", "no_zero_e5_grad_released", "pos_zero_e5_grad2",
            "fraction_zeros_e5_released", "no_zero_e7_grad2_09avg", "no_zero_e5_grad2_09avg",
            "no_zero_e7_grad_released_09avg", "no_zero_e5_grad_released_09avg",
            "fraction_zeros_e7_released_09avg", "fraction_zeros_e5_released_09avg",
            "pos_zero_e7_grad2_09avg", "pos_zero_e5_grad2_09avg", "pos_grad_0_0_e7",
            "pos_grad_0_0_e5", "w_diff_0_0_e7", "w_diff_0_0_e5", "w2_val_0_0_e7",
            "w2_val_0_0_e5", "g2_val_0_0_e7", "g2_val_0_0_e5", "w2_val_0_0_e7_09avg",
            "w2_val_0_0_e5_09avg", "g2_val_0_0_e7_09avg", "g2_val_0_0_e5_09avg",
            "no_grad_flipped_e7", "no_grad_flipped_e5", "fraction_grad_flipped_e7",
            "fraction_grad_flipped_e5", "fraction_grad_flipped_e7_09avg",
            "fraction_grad_flipped_e5_09avg", "pos_min_max_0_0_e7", "no_min_e7",
            "no_max_e7", "no_min_e7_09avg", "no_max_e7_09avg", "pos_min_max_0_0_e5",
            "no_min_e5", "no_max_e5", "no_min_e5_09avg", "no_max_e5_09avg"
        ]
        epoch_wise_keys = ["train_mean_loss", "val_mean_loss", "train_mean_acc", "val_mean_acc"]
        return {
            "batch_wise": {key: [] for key in batch_wise_keys},
            "epoch_wise": {key: [] for key in epoch_wise_keys}
        }

    def _calculate_dummy_grads(self):
        x, y = next(self._train_generator)
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(x)
            loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
        grad = tape.gradient(loss, self.model.trainable_variables)
        # Dummy second order grad
        sec_grad = [tf.zeros_like(g) for g in grad]
        return grad, sec_grad

    def load_model_and_results(self):
        loaded_model = utils.get_stored_result(self._load_model_path, type_="keras")
        if loaded_model:
            print(f"Loaded model from {self._load_model_path}")
            self.model = loaded_model
        else:
            self.model = build_model(config.INPUT_SHAPE, config.NUM_CLASSES)

        loaded_results = utils.get_stored_result(self._load_interim_results_path, type_="json")
        if loaded_results:
            print(f"Loaded interim results from {self._load_interim_results_path}")
            self.interim_results = loaded_results

    def store_model_and_results(self, store_model_path, store_interim_results_path):
        utils.save_result(self.variables['model'], store_model_path, type_="keras")
        utils.save_result(self.interim_results, store_interim_results_path, type_="json")

    def reset_pbar(self):
        self.pbar = tqdm(total=self.t_steps_per_epoch, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

    def initialize_logs_directory(self):
        if os.path.exists(config.LOGS_DIR):
            shutil.rmtree(config.LOGS_DIR)
        print('Initializing the summary writer')
        for result_key, results in self.interim_results.items():
            self.summary_writer[result_key] = {}
            for key in results.keys():
                self.summary_writer[result_key][key] = tf.summary.create_file_writer(f"{config.LOGS_DIR}/{result_key}/{key}")

    def update_logs_directory(self):
        if not self.summary_writer:
            self.initialize_logs_directory()
        for result_key, results in self.interim_results.items():
            for key, items in results.items():
                if any(k in key for k in ['hist', 'pos', 'w_diff', '2_val_']):
                    continue
                values = [np.mean(item) for item in items] if 'batch' in result_key.lower() else items
                writer = self.summary_writer[result_key][key]
                with writer.as_default():
                    for epoch, value in enumerate(values):
                        tf.summary.scalar(key, value, step=epoch)

    def on_train_batch_end(self, batch, logs=None):
        self.variables["g_prev"] = self.variables["g_cur"]
        self.variables["g_cur"] = utils.flatten_grad(logs["grads"])
        self.variables["g_prev_09avg"] = self.variables["g_cur_09avg"]
        self.variables["g_cur_09avg"] = (0.9 * np.array(self.variables["g_cur_09avg"]) + 0.1 * np.array(self.variables["g_cur"])).tolist()
        self.variables["w_prev"] = self.variables["w_cur"]
        self.variables["w_cur"] = utils.flatten_grad(logs['weights'])
        self.variables["w_prev_09avg"] = self.variables["w_cur_09avg"]
        self.variables["w_cur_09avg"] = (0.9 * np.array(self.variables["w_cur_09avg"]) + 0.1 * np.array(self.variables["w_cur"])).tolist()
        self.variables["model"] = self.model.model

        self._update_batch_metrics(batch, logs)
        self._update_pbar(logs)

    def _update_batch_metrics(self, batch, logs):
        metrics = {}
        thresholds = {'e7': (-1e-7, 1e-7), 'e5': (-1e-5, 1e-5)}

        for name, th in thresholds.items():
            (metrics[f'no_zero_{name}_grad1'], metrics[f'no_zero_{name}_grad2'],
             metrics[f'no_zero_{name}_grad_retained'], metrics[f'no_zero_{name}_grad_released'],
             metrics[f'pos_zero_{name}_grad1'], metrics[f'pos_zero_{name}_grad2'],
             metrics[f'pos_grad_0_0_{name}'], metrics[f'w_diff_0_0_{name}'],
             metrics[f'w2_val_0_0_{name}'], metrics[f'g2_val_0_0_{name}'],
             metrics[f'no_grad_flipped_{name}'], metrics[f'pos_min_max_0_0_{name}'],
             metrics[f'no_min_{name}'], metrics[f'no_max_{name}']) = utils.positional_difference(
                self.variables["g_prev"], self.variables["g_cur"],
                self.variables["w_prev"], self.variables["w_cur"], threshold=th)

            (metrics[f'no_zero_{name}_grad1_09avg'], metrics[f'no_zero_{name}_grad2_09avg'],
             metrics[f'no_zero_{name}_grad_retained_09avg'], metrics[f'no_zero_{name}_grad_released_09avg'],
             _, metrics[f'pos_zero_{name}_grad2_09avg'], _, _,
             metrics[f'w2_val_0_0_{name}_09avg'], metrics[f'g2_val_0_0_{name}_09avg'],
             metrics[f'no_grad_flipped_{name}_09avg'], _,
             metrics[f'no_min_{name}_09avg'], metrics[f'no_max_{name}_09avg']) = utils.positional_difference(
                self.variables["g_prev_09avg"], self.variables["g_cur_09avg"],
                self.variables["w_prev_09avg"], self.variables["w_cur_09avg"], threshold=th)

            metrics[f'fraction_zeros_{name}_released'] = metrics[f'no_zero_{name}_grad_released'] / (metrics[f'no_zero_{name}_grad1'] + 1e-9)
            metrics[f'fraction_zeros_{name}_released_09avg'] = metrics[f'no_zero_{name}_grad_released_09avg'] / (metrics[f'no_zero_{name}_grad1_09avg'] + 1e-9)
            metrics[f'fraction_grad_flipped_{name}'] = metrics[f'no_grad_flipped_{name}'] / (metrics[f'no_zero_{name}_grad_retained'] + 1e-9)
            metrics[f'fraction_grad_flipped_{name}_09avg'] = metrics[f'no_grad_flipped_{name}_09avg'] / (metrics[f'no_zero_{name}_grad_retained_09avg'] + 1e-9)

        if batch % 30 == 0:
            self.interim_results['batch_wise']['g_cur_hist'][-1].append(utils.grad_to_hist(logs["grads"], bin_count=self.bin_count, clip=self.clip))
            for name in ['e7', 'e5']:
                self.interim_results['batch_wise'][f'pos_zero_{name}_grad2'][-1].append(metrics[f'pos_zero_{name}_grad2'])
                self.interim_results['batch_wise'][f'pos_zero_{name}_grad2_09avg'][-1].append(metrics[f'pos_zero_{name}_grad2_09avg'])
                self.interim_results['batch_wise'][f'pos_grad_0_0_{name}'][-1].append(metrics[f'pos_grad_0_0_{name}'])
                self.interim_results['batch_wise'][f'w_diff_0_0_{name}'][-1].append(metrics[f'w_diff_0_0_{name}'])
                self.interim_results['batch_wise'][f'w2_val_0_0_{name}'][-1].append(metrics[f'w2_val_0_0_{name}'])
                self.interim_results['batch_wise'][f'g2_val_0_0_{name}'][-1].append(metrics[f'g2_val_0_0_{name}'])
                self.interim_results['batch_wise'][f'w2_val_0_0_{name}_09avg'][-1].append(metrics[f'w2_val_0_0_{name}_09avg'])
                self.interim_results['batch_wise'][f'g2_val_0_0_{name}_09avg'][-1].append(metrics[f'g2_val_0_0_{name}_09avg'])
                self.interim_results['batch_wise'][f'pos_min_max_0_0_{name}'][-1].append(metrics[f'pos_min_max_0_0_{name}'])

        self.interim_results['batch_wise']['train_loss'][-1].append(logs['train_loss'])
        for key, value in metrics.items():
            if key in self.interim_results['batch_wise']:
                self.interim_results['batch_wise'][key][-1].append(value)

    def _update_pbar(self, logs):
        desc = f"Train Loss: {logs['train_loss']:.4f}"
        # Add more details to pbar description if needed
        self.pbar.set_description(desc)
        self.pbar.update()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch += 1
        print(f"\nEpoch {self.epoch} started")
        for key in self.interim_results['batch_wise']:
            self.interim_results['batch_wise'][key].append([])

    def on_epoch_end(self, epoch, logs=None):
        self.interim_results['epoch_wise']['train_mean_loss'].append(logs['train_mean_loss'])
        self.interim_results['epoch_wise']['val_mean_loss'].append(logs['val_mean_loss'])
        self.interim_results['epoch_wise']['train_mean_acc'].append(logs['train_mean_acc'])
        self.interim_results['epoch_wise']['val_mean_acc'].append(logs['val_mean_acc'])

        print("\n")
        for key, val in self.interim_results['epoch_wise'].items():
            print(f'{key}: {val[-1]:.4f}')
        print("\nLatest Gradient Stats:")
        print(utils.calculate_stats(self.variables['g_cur']))
        print('Statistics of the latest weight difference, when the current and the previous gradients become 0')
        print('e7:', utils.calculate_stats(self.interim_results['batch_wise']['w_diff_0_0_e7'][-1][-1]))
        print('e5:', utils.calculate_stats(self.interim_results['batch_wise']['w_diff_0_0_e5'][-1][-1]))
        print("\n")

        self.update_logs_directory()

        if self.save_check_periodically and (self.epoch) % self.period == 0:
            self.store_model_and_results(
                store_model_path=f"{self.periodic_save_dir_path}/{config.PROG_NAME}.keras",
                store_interim_results_path=f"{self.periodic_save_dir_path}/{config.PROG_NAME}.json"
            )
            log_text = f"""
-- Checkpoint Stored --
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
Epoch (1 based): {self.epoch},
Model saved at {self.periodic_save_dir_path}/{config.PROG_NAME}.keras,
Metadata saved at {self.periodic_save_dir_path}/{config.PROG_NAME}.json
--         --          --
"""
            print(log_text)
            utils.save_result(log_text, f"{self.periodic_save_dir_path}/{config.PROG_NAME}.log", "log")

        self.pbar.close()
        self.reset_pbar()
