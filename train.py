import torch
from models.network import CIFAR10Net
from utils.data_loader import CIFAR10DataLoader
from utils.trainer import Trainer
from torchsummary import summary

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = CIFAR10Net().to(device)
    
    # Print model summary
    print("\nModel Summary:")
    summary(model, input_size=(3, 32, 32))
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Initialize data loaders
    data_loader = CIFAR10DataLoader(batch_size=128)
    train_loader, test_loader = data_loader.get_dataloader()
    
    # Initialize trainer
    trainer = Trainer(model, device)
    
    # Training loop
    best_acc = 0
    for epoch in range(40):
        print(f"\nEpoch: {epoch}")
        train_acc = trainer.train(train_loader)
        test_acc = trainer.test(test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f"Best accuracy so far: {best_acc:.2f}%")

if __name__ == "__main__":
    main() 