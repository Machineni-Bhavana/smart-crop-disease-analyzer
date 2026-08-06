from src.utils.logger import get_logger

logger = get_logger("EarlyStopping")

class EarlyStopping:
    """
    Early stopping handler to terminate training when validation loss stops improving.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if validation loss improved. Return True if new best loss achieved.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best loss: {self.best_loss:.4f}, Current loss: {val_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered!")
            return False
