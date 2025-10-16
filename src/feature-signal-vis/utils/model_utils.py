import torch
import torch.nn as nn
import os

def load_model(num_lab, model_class, path=None, device='cpu'):
  """
  Loads a pre-trained model if a path is provided, otherwise creates a new model.

  Args:
      num_lab (int): The number of labels/classes in the dataset.
      model_class (nn.Module): The model class to instantiate.
      path (str, optional): The path to the pre-trained model file. Defaults to None.
      device (str): The device to load the model onto.

  Returns:
      nn.Module: The loaded or newly created model.
  """
  # Create an instance of the CustomModel
  model = model_class(num_classes=num_lab)

  # If a path is provided and the file exists, load the pre-trained weights
  if path and os.path.exists(path):
      model.load_state_dict(torch.load(path, map_location=device))
      print(f"Model loaded from {path}")
  else:
      print(f"Model not found at {path}, creating a new one.")

  # Move the model to the specified device (e.g., CPU or GPU)
  model = model.to(device)
  return model

def save_model(model, path):
  """
  Saves the model's state dictionary to the specified path.

  Args:
      model (nn.Module): The model to be saved.
      path (str): The path to save the model file.
  """
  # Save the model's state dictionary
  torch.save(model.state_dict(), path)
