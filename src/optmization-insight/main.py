from .trainer import Trainer

def main():
    """
    Main function to run the training and analysis.
    """
    # The original script uses google.colab, which is not available here.
    # The paths in config.py point to '/content/...', which is specific to Colab.
    # You might need to adjust the paths in config.py to run this locally.
    print("Starting optimization insight analysis...")
    print("Please ensure that the dataset is available at the path specified in config.py.")
    print("If running outside of Google Colab, you may need to adjust paths in 'src/optmization-insight/config.py'.")
    
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main()
