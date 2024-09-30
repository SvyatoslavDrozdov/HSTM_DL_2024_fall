import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_data(train_csv, val_csv, test_csv):
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)
    test_data = pd.read_csv(test_csv)

    X_train = train_data.drop(columns=['order0', 'order1', 'order2']).values
    y_train = train_data['order0'].values
    X_val = val_data.drop(columns=['order0', 'order1', 'order2']).values
    y_val = val_data['order0'].values
    X_test = test_data.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test


def init_model(input_size, num_classes, lr):
    model = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)

        outputs = model(X_tensor)

        _, predictions = torch.max(outputs, 1)

        accuracy = (predictions.numpy() == y).mean()

        conf_matrix = confusion_matrix(y, predictions.numpy())

    return predictions.numpy(), accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()

        outputs = model(X_train_tensor)

        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_predictions, val_accuracy, val_conf_matrix = evaluate(model, X_val, y_val)

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Validation Accuracy: {val_accuracy:.4f}')
    return model


def main(args):
    X_train, y_train, X_val, y_val, X_test = load_data(args.train_csv, args.val_csv, args.test_csv)

    input_size = X_train.shape[1]
    num_classes = 3
    model, criterion, optimizer = init_model(input_size, num_classes, args.lr)

    model = train(model, criterion, optimizer, X_train, y_train, X_val, y_val, args.num_epoches)

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_outputs = model(X_test_tensor)
        _, test_predictions = torch.max(test_outputs, 1)

    submission_df = pd.DataFrame({'Predicted': test_predictions.numpy()})
    submission_df.to_csv(args.out_csv, index=False)

    print(f'Submission saved to {args.out_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='data/train.csv')
    parser.add_argument('--val_csv', default='data/val.csv')
    parser.add_argument('--test_csv', default='data/test.csv')
    parser.add_argument('--out_csv', default='data/submission.csv')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_epoches', default=1500)

    parameters = parser.parse_args()
    main(parameters)
