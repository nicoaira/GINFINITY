class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in monitored quantity to qualify as improvement.
                            A decrease of more than min_delta counts as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:  # Changed comparison
            # Significant improvement found
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict().copy()
            self.counter = 0  # Reset counter on improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

def train_model_with_early_stopping(
        model,
        model_id,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs,
        patience,
        device,
        log_path,
        decay_rate=0.1  # Add decay_rate parameter
):
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase...
        # [existing training code remains the same]

        # Apply learning rate decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            
        avg_val_loss = val_loss / len(val_loader)
        
        # Log epoch information including learning rate
        current_lr = optimizer.param_groups[0]['lr']
        epoch_log = {
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Training Loss": f"{running_loss / len(train_loader)}",
            "Validation Loss": f"{avg_val_loss}",
            "Learning Rate": f"{current_lr}"
        }
        log_information(log_path, epoch_log)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {avg_val_loss}, Learning Rate: {current_lr}")

        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Restore best weights
    early_stopping.restore_best_weights(model)
    print(f"Restored best model with validation loss: {early_stopping.best_loss}")
    
    finished_reason = "Early stopping" if early_stopping.early_stop else f"{epoch+1} epochs"
    log_information(log_path, {
        "Training finished": finished_reason,
        "Best validation loss": f"{early_stopping.best_loss}"
    })
    print("Training complete.")

    # Save the best model
    save_model_to_local(model, optimizer, epoch, model_id, log_path)

def save_model_to_local(model, optimizer, epoch, model_id, log_path):
    """Save model with metadata to local disk"""
    save_path = f'saved_model/{model_id}.pth'
    model.save_checkpoint(save_path, optimizer, epoch)
    log_information(log_path, {"Model saved": save_path})