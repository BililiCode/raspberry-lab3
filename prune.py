import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os
import time

# import TORCH.NN.UTILS.PRUNE.REMOVE as remove
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device: {device}")

# Hyper-parameters-
# input_size = 784    # 28 x 28, flattened to be 1-D tensor
# hidden_size = 100
num_classes = 10
num_epochs = 1
batch_size = 32
learning_rate = 0.0012

# MNIST dataset statistics:
# mean = tensor([0.1307]) & std dev = tensor([0.3081])
mean = np.array([0.1307])
std_dev = np.array([0.3081])

transforms_apply = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std_dev)
])
# MNIST dataset-
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True,
    transform=transforms_apply, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False,
    transform=transforms_apply
)

# Create dataloader-
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size,
    shuffle=False
)


class LeNet5(nn.Module):
    '''
    Implements a variation of LeNet-5 CNN. It is LeNet-4.
    '''

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6,
            kernel_size=3, padding=1,
            stride=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16,
            kernel_size=3, padding=1,
            stride=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120,
            kernel_size=3, padding=1,
            stride=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )

        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(in_features = 512, out_features = 256)
        # self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        # self.op = nn.Linear(in_features = 84, out_features = 10)
        self.op = nn.Linear(in_features=1080, out_features=10)

        self.weights_initialization()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        return self.op(x)

    def shape_computation(self, x):
        x = self.conv1(x)
        print(f"conv1.shape = {x.shape}")

        x = self.pool(x)
        print(f"pool.shape = {x.shape}")

        x = self.conv2(x)
        print(f"conv2.shape = {x.shape}")

        x = self.pool(x)
        print(f"pool.shape = {x.shape}")

        x = self.conv3(x)
        print(f"conv3.shape = {x.shape}")

        x = self.pool(x)
        print(f"pool.shape = {x.shape}")

        x = self.flatten(x)
        print(f"flatten.shape = {x.shape}")

        x = self.op(x)
        print(f"output.shape = {x.shape}")

    def weights_initialization(self):
        '''
        When we define all the modules such as the layers in '__init__()'
        method above, these are all stored in 'self.modules()'.
        We go through each module one by one. This is the entire network,
        basically.
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


def count_params(model):
    tot_params = 0
    for layer_name, param in model.named_parameters():
        # print(f"{layer_name}.shape = {param.shape} has {torch.count_nonzero(param.data)} non-zero params")
        tot_params += torch.count_nonzero(param.data)

    return tot_params


def train_model(model, train_loader):
    '''
    Function to perform one epoch of training by using 'train_loader'.
    Returns loss and number of correct predictions for this epoch.
    '''
    running_loss = 0.0
    running_corrects = 0.0

    for batch, (images, labels) in enumerate(train_loader):
        # Reshape image and place it on GPU-
        # images = images.reshape(-1, input_size).to(device)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # forward pass
        J = loss(outputs, labels)  # compute loss
        optimizer.zero_grad()  # empty accumulated gradients
        J.backward()  # perform backpropagation
        optimizer.step()  # update parameters

        # Compute model's performance statistics-
        running_loss += J.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        running_corrects += torch.sum(predicted == labels.data)

        '''
        # Print information every 100 steps-
        if (batch + 1) % 100 == 0:
            print(f"epoch {epoch + 1}/{num_epochs}, step {batch + 1}/{num_steps}, loss = {J.item():.4f}")
        '''

    return running_loss, running_corrects


def test_model(model, test_loader):
    total = 0.0
    correct = 0.0
    running_loss_val = 0.0
    latency = 0.0

    with torch.no_grad():

        for images, labels in test_loader:
            # Place features (images) and targets (labels) to GPU-
            # images = images.reshape(-1, input_size).to(device)
            images = images.to(device)
            labels = labels.to(device)
            # print(f"images.shape = {images.shape}, labels.shape = {labels.shape}")

            # Set model to evaluation mode-
            model.eval()

            # Make predictions using trained model
            x = time.time()
            outputs = model(images)
            x1 = time.time()
            latency += (x1 - x)

            _, y_pred = torch.max(outputs, 1)

            # Compute validation loss-
            # J_val = loss(outputs, labels)

            # running_loss_val += J_val.item() * labels.size(0)

            # Total number of labels-
            total += labels.size(0)

            # Total number of correct predictions-
            correct += (y_pred == labels).sum()


    print(latency)

    return (running_loss_val, correct, total)


def compute_sparsity(best_model):
    conv1_sparsity = (torch.sum(best_model.conv1.weight == 0) / best_model.conv1.weight.nelement()) * 100
    conv2_sparsity = (torch.sum(best_model.conv2.weight == 0) / best_model.conv2.weight.nelement()) * 100
    conv3_sparsity = (torch.sum(best_model.conv3.weight == 0) / best_model.conv3.weight.nelement()) * 100
    op_sparsity = (torch.sum(best_model.op.weight == 0) / best_model.op.weight.nelement()) * 100

    # Compute global sparsity-
    num = torch.sum(best_model.conv1.weight == 0) + torch.sum(best_model.conv2.weight == 0) + torch.sum(
        best_model.conv3.weight == 0) \
          + torch.sum(best_model.op.weight == 0)

    denom = best_model.conv1.weight.nelement() + best_model.conv2.weight.nelement() + best_model.conv3.weight.nelement() \
            + best_model.op.weight.nelement()

    global_sparsity = num / denom * 100

    '''
    print(f"conv1.weight has {conv1_sparsity:.2f}% sparsity")
    print(f"conv2.weight has {conv2_sparsity:.2f}% sparsity")
    print(f"conv3.weight has {conv3_sparsity:.2f}% sparsity")
    print(f"op.weight has {op_sparsity:.2f}% sparsity")

    print(f"LeNet-4 Global Sparsity = {global_sparsity:.2f}%")
    '''

    # return None
    return global_sparsity


'''
Iterative layer-wise Pruning algorithm:
Take an already trained model and repeat the following steps x times-

prune p% of smallest magnitude weights in a structured layer-wise manner

fine-tune pruned model to recover performance
'''

best_model = LeNet5().to(device=device)
best_model.load_state_dict(torch.load(
    "./LeNet-4_Trained.pth"))

loss = nn.CrossEntropyLoss()  # applies softmax for loss function
optimizer = torch.optim.Adam(best_model.parameters(), lr=learning_rate)

print(f"number of non-zero parameter in unpruned LeNet-4 CNN = {count_params(best_model)}")

running_loss_val, correct, total = test_model(best_model, test_loader)

# Compute validation loss and accuracy-
val_loss = running_loss_val / len(test_dataset)
val_acc = 100 * (correct / total)

print("Best trained LeNet-4 metrics on validation dataset:")
print(f"val_loss = {val_loss:.4f} & val_acc = {val_acc:.2f}%")

# User input parameters for Early Stopping in manual implementation-
minimum_delta = 0.001
patience = 3

# Initialize parameters for Early Stopping manual implementation-
best_val_loss = 100
loc_patience = 0
history = {}
state = 'training'  # if you want see the inference latency of pruned, please change the 'training' to 'inference'

start = time.time()
running_loss_val, correct, total = test_model(best_model, test_loader)
end = time.time()
latency = end - start

print("latency:", latency)
########iterative pruning the model


for iter_prune_round in range(10):
    print(f"\n\nIterative Global pruning round = {iter_prune_round + 1}")

    # Prune layer-wise in a structured manner-
    prune.ln_structured(best_model.conv1, name="weight", amount=0.1, n=2, dim=0)  # amount is the pruning ratio
    prune.ln_structured(best_model.conv2, name="weight", amount=0.1, n=2, dim=0)
    prune.ln_structured(best_model.conv3, name="weight", amount=0.1, n=2, dim=0)
    prune.ln_structured(best_model.op, name="weight", amount=0.1, n=2, dim=0)

    # Print current global sparsity level-
    print(f"LeNet-5 global sparsity = {compute_sparsity(best_model):.2f}%")

    # Fine-training loop-
    print("\nFine-tuning pruned model to recover model's performance\n")

    # Initialize parameters for Early Stopping manual implementation-
    best_val_loss = 100
    loc_patience = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0.0

        if loc_patience >= patience:
            print("\n'EarlyStopping' called!\n")
            break

        running_loss, running_corrects = train_model(best_model, train_loader)

        # Compute training loss and accuracy for one epoch-
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        running_loss_val, correct, total = test_model(best_model, test_loader)

        # Compute validation loss and accuracy
        epoch_val_loss = running_loss_val / len(test_dataset)
        val_acc = 100 * (correct / total)

        print(
            f"\nepoch: {epoch + 1} training loss = {epoch_loss:.4f}, training accuracy = {epoch_acc * 100:.2f}%, val_loss = {epoch_val_loss:.4f} & val_accuracy = {val_acc:.2f}%\n")

        # Code for manual Early Stopping:
        if (epoch_val_loss < best_val_loss) and np.abs(epoch_val_loss - best_val_loss) >= minimum_delta:

            # update 'best_val_loss' variable to lowest loss encountered so far-
            best_val_loss = epoch_val_loss

            # reset 'loc_patience' variable
            loc_patience = 0

            print(
                f"Saving model with lowest val_loss = {epoch_val_loss:.4f} for iterative pruning round = {iter_prune_round + 1}\n")

            # Save trained model with validation accuracy-
            torch.save(best_model.state_dict(), f"LeNet-4_Trained_sparsity-{compute_sparsity(best_model):.2f}.pth")

            # Save 'best' model's metrics to Python3 dict-
            history[iter_prune_round + 1] = {'sparsity': compute_sparsity(best_model), 'best_acc': epoch_acc * 100,
                                             'best_loss': epoch_loss,
                                             'best_val_acc': val_acc, 'best_val_loss': epoch_val_loss}

        else:  # there is no improvement in monitored metric 'val_loss'
            loc_patience += 1  # number of epochs without any improvement

    # if you want to get the latency of pruned model for the inference stage, please remove the comment.
    #  reference: https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html#torch.nn.utils.prune.remove
    if state == 'inference':
        start = time.time()
        best_model.conv1 = prune.remove(best_model.conv1, name='weight')
        best_model.conv2 = prune.remove(best_model.conv2, name='weight')
        best_model.conv3 = prune.remove(best_model.conv3, name='weight')
        best_model.op = prune.remove(best_model.op, name='weight')

        _, _, _ = test_model(best_model, test_loader)
        end = time.time()
        latency = end - start

        print("after pruning latency:", latency)

####Visualize training metrics:
# Python3 dict for training accuracy-
plot_acc = {}
for epoch in history.keys():
    sparsity = history[epoch]['sparsity'].item()
    plot_acc[sparsity] = history[epoch]['best_acc'].item()

# Python3 dict for validation accuracy-
plot_val_acc = {}
for epoch in history.keys():
    sparsity = history[epoch]['sparsity'].item()
    plot_val_acc[sparsity] = history[epoch]['best_val_acc'].item()



for epoch in history.keys():
    print(
        f"sparsity = {history[epoch]['sparsity'].item():.2f}%, best acc = {history[epoch]['best_acc']:.2f}% & best val acc = {history[epoch]['best_val_acc'].item():.2f}%")
