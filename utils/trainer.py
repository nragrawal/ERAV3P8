import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        total_steps = 30 * 391  # 40 epochs * steps_per_epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=30,  # Increased epochs
            steps_per_epoch=391,
            pct_start=0.3,  # Modified warm-up period
            div_factor=10,
            final_div_factor=100,
            anneal_strategy='cos'  # Added cosine annealing
        )

    def train(self, train_loader):
        self.model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            pred = self.model(data)
            loss = self.criterion(pred, target)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            
            pbar.set_description(
                desc=f'Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
            )
        
        return 100*correct/processed

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return accuracy 