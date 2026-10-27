import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from collections import Counter
from models.custom_model import CustomNet


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        # compute prediction and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

if __name__ == "__main__":
    transform = T.Compose([
        T.Resize((64, 64)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)
    print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
    print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)

    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)

        # Best validation accuracy
        if best_acc < val_accuracy:
            best_acc = val_accuracy
            torch.save(model.state_dict(), "checkpoints/custom_model_weights.pth")    

    print(f'Best validation accuracy: {best_acc:.2f}%')
