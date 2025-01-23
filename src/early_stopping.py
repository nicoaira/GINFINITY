class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
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
            self.best_state_dict = model.state_dict().copy()  # Save initial state
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict().copy()  # Save best state
            self.counter = 0

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
        log_path
):
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase...
        # [existing training code remains the same]

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
        
        epoch_log = {
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Training Loss": f"{running_loss / len(train_loader)}",
            "Validation Loss": f"{avg_val_loss}"
        }
        log_information(log_path, epoch_log)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {avg_val_loss}")

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