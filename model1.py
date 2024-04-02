import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data loading and transformation
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.USPS(root='data/', train=True, transform=transform, download=True)
test_dataset = datasets.USPS(root='data/', train=False, transform=transform, download=True)


train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
def perform_eda(dataset):
   plt.figure(figsize=(10, 8))
   for i in range(9):
       img, label = dataset[i]
       img = img.numpy().squeeze()
       plt.subplot(3,3,i+1)
       plt.imshow(img, cmap='gray')
       plt.title(f'Label: {label}')
       plt.axis('off')
   plt.show()


perform_eda(train_dataset)
class MLP(nn.Module):
   def __init__(self):
       super(MLP, self).__init__()
       self.flatten = nn.Flatten()
       self.fc_layers = nn.Sequential(
           nn.Linear(16*16, 512),
           nn.ReLU(),
           nn.Linear(512, 256),
           nn.ReLU(),
           nn.Linear(256, 10)
       )


   def forward(self, x):
       x = self.flatten(x)
       return self.fc_layers(x)
class CNN(nn.Module):
   def __init__(self):
       super(CNN, self).__init__()
       self.conv_layers = nn.Sequential(
           nn.Conv2d(1, 32, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2, 2),
           nn.Conv2d(32, 64, kernel_size=3, padding=1),
           nn.ReLU()
       )
       self.fc_layers = nn.Sequential(
           nn.Linear(64 * 8 * 8, 128),
           nn.ReLU(),
           nn.Linear(128, 10)
       )


   def forward(self, x):
       x = self.conv_layers(x)
       x = x.view(-1, 64 * 8 * 8)
       return self.fc_layers(x)
def log_pr_curve(writer, phase, model, data_loader, epoch, num_classes=10):
   model.eval()
   all_preds = torch.tensor([])
   all_probs = torch.tensor([])
   all_targets = torch.tensor([])
   with torch.no_grad():
       for data, target in data_loader:
           data, target = data.to(device), target.to(device)
           output = model(data)
           probs = torch.softmax(output, dim=1)
           all_preds = torch.cat((all_preds, output.argmax(dim=1).cpu()), dim=0)
           all_probs = torch.cat((all_probs, probs.cpu()), dim=0)
           all_targets = torch.cat((all_targets, target.cpu()), dim=0)
   for i in range(num_classes):
       labels_i = all_targets == i
       probs_i = all_probs[:, i]
       writer.add_pr_curve(f'{phase}/Precision_recall_class_{i}', labels_i, probs_i, epoch)
def train_model(model, train_loader, optimizer, criterion, epochs, model_name='Model', writer=None):
   model.train()
   for epoch in range(epochs):
       total_loss = 0
       for data, target in train_loader:
           data, target = data.to(device), target.to(device)
           optimizer.zero_grad()
           output = model(data)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()
           total_loss += loss.item()
       if writer:
           writer.add_scalar(f'{model_name}/Loss', total_loss/len(train_loader), epoch)
from sklearn.metrics import precision_recall_curve
from torch.nn.functional import softmax


def evaluate_model(model, data_loader, criterion, model_name='Model', writer=None, epoch=0):
   model.eval()
   test_loss = 0
   correct = 0
   all_preds = []
   all_targets = []
   probs_list = []
   with torch.no_grad():
       for data, target in data_loader:
           data, target = data.to(device), target.to(device)
           output = model(data)
           probs = softmax(output, dim=1)
           test_loss += criterion(output, target).item()
           pred = output.argmax(dim=1, keepdim=True)
           correct += pred.eq(target.view_as(pred)).sum().item()
           all_preds.extend(pred.view(-1).cpu().numpy())
           all_targets.extend(target.view(-1).cpu().numpy())
           probs_list.append(probs.cpu().numpy())
   test_loss /= len(data_loader.dataset)
   accuracy = 100. * correct / len(data_loader.dataset)
   precision = precision_score(all_targets, all_preds, average='macro')
   recall = recall_score(all_targets, all_preds, average='macro')
   conf_matrix = confusion_matrix(all_targets, all_preds)
   all_probs = np.concatenate(probs_list)


   if writer:
       writer.add_scalar(f'{model_name}/Accuracy', accuracy, epoch)
       writer.add_scalar(f'{model_name}/Precision', precision, epoch)
       writer.add_scalar(f'{model_name}/Recall', recall, epoch)
       for i in range(all_probs.shape[1]):
           labels_i = (np.array(all_targets) == i).astype(int)
           probs_i = all_probs[:, i]
           precision_i, recall_i, _ = precision_recall_curve(labels_i, probs_i)
           writer.add_pr_curve(f'{model_name}/PR_curve_class_{i}', labels_i, probs_i, epoch)


   return test_loss, accuracy, precision, recall, conf_matrix


def main():
   writer = SummaryWriter()


   mlp = MLP().to(device)
   cnn = CNN().to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer_mlp = optim.Adam(mlp.parameters())
   optimizer_cnn = optim.Adam(cnn.parameters())


   epochs = 10


   for epoch in range(epochs):
       train_model(mlp, train_loader, optimizer_mlp, criterion, 1, 'MLP', writer)
       metrics_mlp = evaluate_model(mlp, test_loader, criterion, 'MLP', writer, epoch)
       print(f"MLP - Epoch {epoch+1}, Loss: {metrics_mlp[0]}, Accuracy: {metrics_mlp[1]}%, Precision: {metrics_mlp[2]}, Recall: {metrics_mlp[3]}")


   for epoch in range(epochs):
       train_model(cnn, train_loader, optimizer_cnn, criterion, 1, 'CNN', writer)
       metrics_cnn = evaluate_model(cnn, test_loader, criterion, 'CNN', writer, epoch)
       print(f"CNN - Epoch {epoch+1}, Loss: {metrics_cnn[0]}, Accuracy: {metrics_cnn[1]}%, Precision: {metrics_cnn[2]}, Recall: {metrics_cnn[3]}")


   writer.close()


if __name__ == "__main__":
   main()
