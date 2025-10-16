import json
import os
from typing import Literal
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def save_result(result, path:str, type_:Literal["json","csv","keras","log"]):
  """Saves results to a file."""
  if type_ == "json":
      with open(path, 'w') as f:
          json.dump(result, f, indent=4)
  elif type_ == "csv":
      if isinstance(result, pd.DataFrame):
          result.to_csv(path, index=False)
      else:
          raise ValueError("The result is not a dataframe")
  elif type_ == "keras":
      result.save(path)
  elif type_ == "log":
      with open(path, 'a') as f:
          f.write(result)
  else:
      raise ValueError("Invalid type")


def get_stored_result(path:str, type_:Literal["json","csv","keras"]):
    """Loads results from a file."""
    if not os.path.exists(path):
        return None
    if type_ == "json":
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    elif type_ == "csv":
        # The original code had json.load for csv, correcting it to pd.read_csv
        return pd.read_csv(path)
    elif type_ == "keras":
        return keras.models.load_model(path, compile=False)
    else:
        return None

def angle_evaluator(gradient1, gradient2):
  """Calculates the angle between two gradients."""
  dot_product=0.0
  for grad1,grad2 in zip(gradient1,gradient2):
    dot_product+=tf.math.reduce_sum(tf.math.multiply(grad1,grad2))
  norm1 = norm_evaluator(gradient1)
  norm2 = norm_evaluator(gradient2)
  if norm1==0.0 or norm2==0.0:
    return 0.0
  return tf.math.acos(tf.math.divide(dot_product,tf.math.multiply(norm1,norm2)))/tf.constant(np.pi, dtype=tf.float32)*180

def norm_evaluator(grad):
  """Calculates the norm of a gradient."""
  norm=0.0
  for grad_ in grad:
    norm+=tf.math.reduce_sum(tf.math.square(grad_))
  norm = tf.math.sqrt(norm)
  return norm

def float_list_to_histogram(data, bin_count=10):
    """Creates a histogram from a list of floats."""
    counts, bin_edges = np.histogram(data, bins=bin_count)
    bin_ranges = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]
    histogram = {str(bin_range): count.item() for bin_range, count in zip(bin_ranges, counts)}
    return histogram

def flatten_grad(grad):
  """Flattens a list of gradient tensors into a single list."""
  grad_flatten=[]
  for grad_ in grad:
    _grad_store=tf.reshape(grad_,[-1]).numpy().tolist()
    grad_flatten.extend(_grad_store)
  return grad_flatten

def grad_to_hist(grad, bin_count=100,clip=(-1e9,1e9)):
  """Converts a gradient to a histogram."""
  grad_flatten=flatten_grad(grad)
  grad_flatten=np.clip(grad_flatten,clip[0],clip[1])
  histogram = float_list_to_histogram(grad_flatten, bin_count=bin_count)
  return histogram

def positional_difference(grad1_f, grad2_f, w1_f, w2_f, threshold=(-1e-9, 1e-9)):
    """Analyzes the positional differences between two sets of gradients and weights."""
    grad1_flatten = np.array(grad1_f)
    grad2_flatten = np.array(grad2_f)
    w1_flatten = np.array(w1_f)
    w2_flatten = np.array(w2_f)

    if threshold == "Zero":
        pos1 = (grad1_flatten != 0.0)
        pos2 = (grad2_flatten != 0.0)
    else:
        pos1 = (grad1_flatten < threshold[0]) | (grad1_flatten > threshold[1])
        pos2 = (grad2_flatten < threshold[0]) | (grad2_flatten > threshold[1])

    non_zero_mask_g1 = pos1
    zero_mask_g1 = ~non_zero_mask_g1

    no_zero_grad1 = np.sum(zero_mask_g1)
    no_zero_grad2 = np.sum(~pos2)
    
    retained_mask = zero_mask_g1 & ~pos2
    released_mask = zero_mask_g1 & pos2

    no_zero_grad_retained = np.sum(retained_mask)
    no_zero_grad_released = np.sum(released_mask)

    pos_zero_grad1 = np.where(zero_mask_g1)[0].tolist()
    pos_zero_grad2 = np.where(~pos2)[0].tolist()
    pos_grad_0_0 = np.where(retained_mask)[0].tolist()

    w_diff_0_0 = np.abs(w2_flatten - w1_flatten)[retained_mask].tolist()
    w2_val_0_0 = w2_flatten[retained_mask].tolist()
    g2_val_0_0 = grad2_flatten[retained_mask].tolist()

    g1g2_sign_flipped = (grad1_flatten * grad2_flatten) < 0
    no_grad_flipped = np.sum(g1g2_sign_flipped & retained_mask)

    # This part seems to have a logic issue in the original code.
    # The multiplication by g1g2_0_0 (g1g2_sign_flipped) doesn't make sense for min/max detection
    # when gradients are zero. I'll keep the logic as it was but it might need review.
    min_max_indicator = np.sign((w2_flatten - w1_flatten) * (grad2_flatten - grad1_flatten))
    pos_min_max_0_0 = (min_max_indicator * g1g2_sign_flipped)[retained_mask]
    
    no_min = np.sum(pos_min_max_0_0 == 1)
    no_max = np.sum(pos_min_max_0_0 == -1)
    pos_min_max_0_0 = pos_min_max_0_0.astype('int32').tolist()

    return (
        no_zero_grad1, no_zero_grad2, no_zero_grad_retained, no_zero_grad_released,
        pos_zero_grad1, pos_zero_grad2, pos_grad_0_0, w_diff_0_0, w2_val_0_0,
        g2_val_0_0, no_grad_flipped, pos_min_max_0_0, no_min, no_max
    )


def calculate_stats(data):
    """Calculate statistics for the given data."""
    if len(data) == 0:
        return None
    data_np = np.array(data)
    stats = {
        "min": np.min(data_np).item(),
        "max": np.max(data_np).item(),
        "mean": np.mean(data_np).item(),
        "median": np.median(data_np).item(),
        "stddev": np.std(data_np, ddof=1).item()
    }
    return stats

def create_stats_dataframe(interim_results):
    """Create a DataFrame for the statistics."""
    data = []
    for result_name in interim_results:
        for key, batch_data in interim_results[result_name].items():
            if not batch_data: continue
            stats = calculate_stats(batch_data[-1])
            if stats:
                data.append([
                    result_name, key, stats['min'], stats['max'],
                    stats['mean'], stats['median'], stats['stddev']
                ])
    df = pd.DataFrame(data, columns=["Result Name", "Key", "Min", "Max", "Mean", "Median", "Stddev"])
    print(df)

def plot_histogram_from_dict(histogram, show=True):
    """Plots a histogram from a dictionary."""
    bin_ranges = [eval(br) for br in histogram.keys()]
    frequencies = list(histogram.values())
    bin_midpoints = [(br[0] + br[1]) / 2 for br in bin_ranges]
    bin_widths = [(br[1] - br[0]) for br in bin_ranges]
    
    plt.bar(bin_midpoints, frequencies, width=bin_widths)
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    if show:
        plt.show()
