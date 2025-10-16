import torch.nn as nn

class CustomLoss(nn.Module):
    """
    Custom loss class combining Cross-Entropy Loss and multiple other losses.
    """
    def __init__(self, xce_loss_factor=1.0):
        """
        Initializes the CustomLoss class.

        Args:
            xce_loss_factor (float, optional): Weighting factor for Cross-Entropy Loss. Defaults to 1.0.
        """
        super(CustomLoss, self).__init__()

        # Initialize Cross-Entropy Loss object
        self.xce_loss_obj = nn.CrossEntropyLoss()

        # Store the weighting factor for Cross-Entropy Loss
        self.xce_loss_factor = xce_loss_factor

    def forward(self, y_true_oh, y_pred):
        """
        Calculates the total loss by doing weighted sum of Cross-Entropy Loss and other losses.

        Args:
            y_true_oh (torch.Tensor): One-hot encoded true labels. Shape : (batch_size, num_classes)
            y_pred (torch.Tensor): Predicted probabilities. Shape : (batch_size, num_classes)

        Returns:
            torch.Tensor: Total loss value. Scalar.
        """
        # Calculate Cross-Entropy Loss
        xce_loss_val = self.xce_loss_obj(y_pred, y_true_oh)

        # Calculate total loss by applying the weighting factor
        total_loss = xce_loss_val * self.xce_loss_factor

        return total_loss
