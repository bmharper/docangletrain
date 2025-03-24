import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from model import TextAngleClassifier  # Import the model from model.py


def get_data_loaders(batch_size):
    # Define transformations to preprocess the images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
        #transforms.Resize((32, 32)),  # Resize images to 32x32 pixels (-- our images are already 32x32)
        transforms.ToTensor(),  # Convert images to tensors (scales to [0, 1])
    ])

    # Load the synthetic training and validation datasets
    synth_train_dataset = datasets.ImageFolder(root='../images/synth/train', transform=transform)
    synth_val_dataset = datasets.ImageFolder(root='../images/synth/val', transform=transform)

    # Load the real training and validation datasets
    real_train_dataset = datasets.ImageFolder(root='../images/real/train', transform=transform)
    real_val_dataset = datasets.ImageFolder(root='../images/real/val', transform=transform)

    train_dataset = torch.utils.data.ConcatDataset([synth_train_dataset, real_train_dataset])
    val_dataset = torch.utils.data.ConcatDataset([synth_val_dataset, real_val_dataset])

    # Create the training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Number of images per batch
        shuffle=True  # Shuffle the data for training
    )

    # Create the validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Same batch size
        shuffle=False  # No shuffling for validation
    )

    return train_loader, val_loader


def train():
    # Hyperparameters
    num_epochs = 500
    learning_rate = 0.001
    batch_size = 32

    # Set device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    train_loader, val_loader = get_data_loaders(batch_size)

    # Initialize model, loss function, and optimizer
    model = TextAngleClassifier().to(device)
    # Suitable for classification with 4 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the learning rate scheduler
    #scheduler = lr_scheduler.ReduceLROnPlateau(
    #    optimizer,
    #    mode='min',  # Monitor a metric we want to minimize (validation loss)
    #    factor=0.1,  # Reduce LR by a factor of 0.1 when triggered
    #    patience=20  # Wait 5 epochs without improvement before reducing LR
    #    #verbose=True  # Print a message when LR is reduced
    #)
    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=num_epochs, anneal_strategy='linear', cycle_momentum=False)

    run = wandb.init(entity="wefi", project="docangle", config={"learning_rate": learning_rate, "epochs": num_epochs, "batch_size": batch_size, "scheduler": "OneCycleLR"})

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()
            scheduler.step()  # For OneCycle
        avg_loss = running_loss / len(train_loader)
        #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation for validation
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)  # Accumulate total loss
                _, predicted = torch.max(outputs.data, 1)  # Get predictions
                total += labels.size(0)  # Total samples
                correct += (predicted == labels).sum().item()  # Correct predictions
        avg_val_loss = val_loss / total  # Average validation loss per sample
        accuracy = 100 * correct / total  # Validation accuracy
        last_lr = optimizer.param_groups[0]['lr']  # Current learning rate
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, LR: {last_lr:.6f}")
        run.log({"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": avg_val_loss, "val_accuracy": accuracy, "learning_rate": last_lr})

        # Step the scheduler based on validation loss
        #scheduler.step(avg_val_loss)

    run.finish()

    # Save the trained model weights
    torch.save(model.state_dict(), "text_angle_classifier.pth")

    # Export to TorchScript for NCNN conversion
    model.eval()
    # Dummy input matching input shape
    example_input = torch.rand(1, 1, 32, 32).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("text_angle_classifier.pt")
    print("Model traced and saved as 'text_angle_classifier.pt'")

    # Instructions for NCNN conversion
    print("\nTo convert the model to NCNN format:")
    print("1. Install PNNX (follow NCNN's PNNX documentation).")
    print("2. Run the following command:")
    print("   pnnx text_angle_classifier.pt inputshape=[1,1,32,32]")
    print("3. Use the generated .param and .bin files with NCNN as per its documentation.")


if __name__ == "__main__":
    train()
