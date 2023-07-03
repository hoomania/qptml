from torch.utils.data import DataLoader
from tqdm import tqdm
import dataset_maker as dsm
import fully_connected as fcm
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


class TrainModel:

    def __init__(self, data_title: str, train_data_path: str, test_data_path, n_node_hidden_layer: int,
                 n_classes: int, n_epoch: int, lr: float, batch_size: int, weight_decay: float):
        self.data_title = data_title
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.save_model_path = fr'./saved_model/model_{data_title}.pth'
        self.n_node_hidden_layer = n_node_hidden_layer
        self.n_classes = n_classes
        self.n_epoch = n_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        feature_length = pd.read_csv(train_data_path)
        self.feature_length = feature_length.shape[1] - 1

    def train_model(self, loss_func: str = 'CrossEntropyLoss') -> None:

        print(f'Training model for {self.data_title} is started. \n')

        dataset = dsm.DatasetMaker(self.train_data_path, self.feature_length)

        model = fcm.FC(input_dim=self.feature_length, output_dim=self.n_classes, hidden_layer=self.n_node_hidden_layer)

        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        criterion = torch.nn.CrossEntropyLoss()
        if loss_func == 'BCELoss':
            criterion = torch.nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.n_epoch):
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (inputs, labels) in loop:
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(inputs)
                # Compute and print loss
                loss = criterion(y_pred, labels)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                # perform a backward pass (backpropagation)
                loss.backward()
                # Update the parameters
                optimizer.step()

                loop.set_description(f"Epoch [{str.zfill(str(epoch + 1), 2)}/{self.n_epoch}]")
                loop.set_postfix(loss=loss.item())

        print(f'Training model for {self.data_title} finished. \nModel is saving...')
        torch.save(model.state_dict(), self.save_model_path)
        print(f'Model saved. \n')

    def test_model(self) -> None:

        dataset = dsm.DatasetMaker(self.test_data_path, self.feature_length)

        model = fcm.FC(input_dim=self.feature_length, output_dim=self.n_classes, hidden_layer=self.n_node_hidden_layer)
        model.load_state_dict(torch.load(self.save_model_path))

        test_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        arr_labels = []
        arr_predicts = []
        arr_pure_predict = []
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(test_loader):
                # calculate output by running through the network
                outputs = model(inputs)
                arr_pure_predict.extend(outputs.data)
                # get the predictions
                __, predicted = torch.max(outputs.data, 1)
                # update results
                arr_predicts.extend(predicted)
                arr_labels.extend(labels)

        # confusion_graph(confusion_matrix(arr_labels, arr_predicts))
        # print(classification_report(arr_labels, arr_predicts, zero_division=True))
        # print(f'accuracy: {accuracy_score(arr_labels, arr_predicts):.2}')
        # print(confusion_matrix(arr_labels, arr_predicts))

        cnvt_predict = []
        for row in arr_pure_predict:
            cnvt_predict.append([float(row[i]) for i in range(self.n_classes)])

        dataframe = pd.DataFrame(cnvt_predict)
        dataframe.to_csv(fr'./predict/predict_{self.data_title}.csv')

    def plot_predict(self) -> None:

        data = pd.read_csv(f'./predict/predict_{self.data_title}.csv', index_col=0)
        y_axis_data = [i for i in range(data.shape[0])]
        labels = ['super fluid', 'matt insulator', 'density wave']
        for j in range(self.n_classes):
            plt.plot(y_axis_data, data.iloc[:, j], marker='o', markersize=3, label=labels[j])

        plt.title('Phase prediction on Extended Bose-Hubbard Model')
        plt.xlabel('u (or v)')
        plt.ylabel('probability')
        plt.legend()
        plt.savefig(f'./diagram/diagram_{self.data_title}.png')
        plt.show()
