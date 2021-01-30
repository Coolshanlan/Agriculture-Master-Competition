# %%
import pandas as pd
from datetime import datetime
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score

# writer = SummaryWriter('~/log')

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


# %%
def loadData():
    trainData = pd.read_csv("IA/train_data.csv", index_col=0)
    testData = pd.read_csv("IA/test_data.csv", index_col=0)

    trainData = trainData.drop('d.rainfall_detect', axis=1)
    testData = testData.drop('d.rainfall_detect', axis=1)

    missing = [i for i, v in enumerate(trainData.loc[:, 'd.wind_speed']) if v == -9999.0]
    for i in missing:
        trainData.loc[i, 'd.wind_speed'] = np.nan
    wind_speed_mean = trainData.loc[:, 'd.wind_speed'].mean()

    missing = [i for i, v in enumerate(testData.loc[:, 'd.wind_speed']) if v == -9999.0]
    for i in missing:
        testData.loc[i, 'd.wind_speed'] = np.nan
    trainData.iloc[:, 0] = trainData.iloc[:, 0].map(lambda x: int(x[11:13]))
    testData.iloc[:, 0] = testData.iloc[:, 0].map(lambda x: int(x[11:13]))

    trainData = trainData.fillna(wind_speed_mean)
    testData = testData.fillna(wind_speed_mean)
    X_data = trainData.iloc[:, :18].to_numpy()
    y_data = trainData.iloc[:, 18:].to_numpy()
    X_test = testData.iloc[:, :18].to_numpy()

    return X_data, y_data, X_test

# %%
def getLoader(x, y, X_test):
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8,
                                                          random_state=int(datetime.now().timestamp()))
    transformer = Normalizer().fit(X_train)
    transformer.transform(X_train)
    transformer.transform(X_valid)
    transformer.transform(X_test)

    train_loader = torch.utils.data.DataLoader(dataset(X_train, y_train, True), batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset(X_valid, y_valid, True), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset(X_test, None, False), batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


# %%
class MLP(nn.Module):
    def __init__(self, inputSize, outputSize, args):
        super(MLP, self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.fc1 = nn.Linear(self.inputSize, args['fc1'])
        self.dp1 = torch.nn.Dropout(args['dp1'])
        self.bn1 = nn.BatchNorm1d(args['fc1'])
        self.fc2 = nn.Linear(args['fc1'], args['fc2'])
        self.dp2 = torch.nn.Dropout(args['dp1'])
        self.bn2 = nn.BatchNorm1d(args['fc2'])
        self.fc3 = nn.Linear(args['fc2'], self.outputSize)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x


class dataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, train=True):
        self.x = x
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.train:
            return torch.as_tensor(self.x[i], dtype=torch.float), torch.as_tensor(self.y[i], dtype=torch.float)
        else:
            return torch.as_tensor(self.x[i], dtype=torch.float)


# %%
def train(model, train_loader, valid_loader, optimizer, criterion, epochs):
    stopCount = 0
    maxAcc = -1
    for e in range(epochs):
        total_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            data, target = data[0].to(device), data[1].to(device)
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss_value = loss.item()
            loss.backward()
            total_loss += loss_value

            _ = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.25)

            optimizer.step()


        # writer.add_scalar('Train Loss', total_loss / (len(train_loader.dataset) // batch_size), e)
        # writer.flush()

        f1 = valid(valid_loader, model, criterion, e)
        if f1 > maxAcc:
            maxAcc = f1
            stopCount = 0
        else:
            stopCount += 1

        if stopCount == 5:
            break
    return f1


# %%
def valid(validLoader, model, criterion, e):
    total_loss = 0
    model.eval()
    logitsHis = []
    targetHis = []
    for i, data in enumerate(validLoader):
        data, target = data[0].to(device), data[1].to(device)
        logits = model(data)
        loss = criterion(logits, target)
        loss_value = loss.item()
        total_loss += loss_value
        logitsHis.extend(logits.greater(0.5).cpu().numpy().reshape(-1, 11))
        targetHis.extend(target.cpu().numpy())
    logitsHis = np.array(logitsHis)
    targetHis = np.array(targetHis)
    f1 = f1_score(logitsHis, targetHis, average='micro', zero_division=True)

    # writer.add_scalar('Test Loss', total_loss / (len(validLoader.dataset) // batch_size), e)
    # writer.add_scalar('Test Accuracy', f1, e)
    # writer.flush()
    return f1


# %%
batch_size = 2048
epochs = 100
X_data, y_data, X_test = loadData()

# %%
space = {
    'fc1': hp.choice('fc1', range(5, 200, 10)),
    'fc2': hp.choice('fc2', range(5, 200, 10)),
    'dp1': hp.uniform('dp1', 0.1, 0.5),
    'dp2': hp.uniform('dp2', 0.1, 0.5)
}


def objective(args):
    train_loader, valid_loader, test_loader = getLoader(X_data, y_data, X_test)
    mlp = MLP(X_data.shape[-1], y_data.shape[-1], args).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), weight_decay=0.01)
    f1 = train(mlp, train_loader, valid_loader, optimizer, criterion, epochs)
    return - f1


# %%
best = fmin(objective, space, algo=tpe.suggest, max_evals=30)
print(best)
print(space_eval(space, best))

# %%

train_loader, valid_loader, test_loader = getLoader(X_data, y_data, X_test)
#%%
mlp = MLP(X_data.shape[-1], y_data.shape[-1], space_eval(space, best)).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(mlp.parameters(), weight_decay=0.01)
f1 = train(mlp, train_loader, valid_loader, optimizer, criterion, epochs)
torch.save(mlp.state_dict(), f'IA/mlp.pt')
# %%
sample = pd.read_csv('IA/submission.csv', index_col=0)
# %%
prediction = []
mlp.eval()
for i, data in tqdm(enumerate(test_loader), total=len(test_loader.dataset) // batch_size):
    data, _ = data[0].to(device), data[1].to(device)
    logits = mlp(data)
    prediction.append(logits.greater(0.5).long().cpu().numpy())

prediction = np.array(prediction).reshape(-1, 11)
# %%
submission = pd.DataFrame(prediction, columns=sample.columns.to_list())
# %%
submission.to_csv('IA/NN.csv')

