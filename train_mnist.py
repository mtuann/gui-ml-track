import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import time
import tqdm
from datetime import datetime

# Define the neural network architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Instantiate the CNN model
model = CNN()

# Define loss function and optimizer


# hyper_params = {
#     "optimizer": "adam",
#     "learning_rate": 0.001,
#     "num_epochs": 5
# }


def training_and_update(train_params, rdb, rdb_item):
    print(f"Training with hyperparameters: {train_params}")
    lr = train_params["learning_rate"]
    optimizer = train_params["optimizer"].lower()
    num_epochs = train_params["num_epochs"]
    
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    
    meta_data = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "time_training": [],
        "epoch": [],
    }
    
    # Training loop
    start_time = time.time()
    
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        training_loss, training_correct, train_total = 0.0, 0.0, 0
        tqdm.tqdm.write(f'Epoch [{epoch+1}/{num_epochs}]')
        tqdm_loader = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, labels) in tqdm_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            num_correct = (torch.argmax(outputs, dim=1) == labels).sum().item()
            training_correct += num_correct
            train_total += len(labels)
            
            # if (i+1) % 100 == 0:
            # tqdm.tqdm.write(f'Training - Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {training_loss/100} Accuracy: {training_correct*100/train_total}%')
            # update progress bar
            tqdm_loader.set_description(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {training_loss/100:.4f} Accuracy: {training_correct*100/train_total:.4f}%')
            
            #     print(f'Training - Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100}')
            #     running_loss = 0.0
        train_accuracy = 100 * training_correct / train_total
        print(f'Training - Epoch [{epoch+1}/{num_epochs}], Accuracy: {train_accuracy:.4f}%')
        # Testing phase
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_accuracy = 100 * test_correct / test_total
        print(f'Testing - Epoch [{epoch+1}/{num_epochs}], Accuracy: {test_accuracy:.4f}%')

        elapsed_time = time.time() - start_time
        meta_data["train_loss"].append(training_loss)
        meta_data["train_accuracy"].append(train_accuracy)
        meta_data["time_training"].append(elapsed_time)
        meta_data["test_loss"].append(test_loss)
        meta_data["test_accuracy"].append(test_accuracy)
        meta_data["epoch"].append(epoch)
        
        rdb_item['in_update'] = meta_data
        status = 'done' if epoch == num_epochs - 1 else 'running'
        rdb_item['status'] = status
        rdb.update(rdb_item)
        
    # rd_item.update({
    #     'status': 'done',
    #     'output_json': meta_data,
    # })
        
    # print('Finished Training')
    return meta_data


# def test_dump_training_data_redis():
#     import redis
#     import json
#     r = redis.StrictRedis(host='localhost', port=6379, db=0)
#     # meta_data = get_training_data(hyper_params)
#     fn = "/home/vishc2/tuannm/gui-ml-track/meta_data__2024-03-10_13-44-48.json"
#     meta_data = json.load(open(fn, 'r'))
    
#     # dump meta_data to a json file with indent=2
#     # import json
#     # fn_datetime = f"meta_data__{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
#     # with open(fn_datetime, 'w') as f:
#     #     json.dump(meta_data, f, indent=2)
#     # print(f'{fn_datetime} has been saved')
#     r.set('meta_data', json.dumps(meta_data))
#     print('meta_data has been saved to redis')
    

if __name__ == "__main__":
    pass
    # get parameters from command line
    # args = sys.argv
    # hyper_params["optimizer"] = args[1]
    # hyper_params["learning_rate"] = float(args[2])
    # hyper_params["num_epochs"] = int(args[3])
    
    # test_dump_training_data_redis()
    
    # sample command: python train_mnist.py adam 0.001 5
    # meta_data = get_training_data(hyper_params)
    
    # dump meta_data to a json file with indent=2
    # import json
    # fn_datetime = f"meta_data__{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    # with open(fn_datetime, 'w') as f:
    #     json.dump(meta_data, f, indent=2)
    # print(f'{fn_datetime} has been saved')
    