
import mlflow,pickle,json,requests
from collections import OrderedDict

from typing import Tuple



def load_data(path):
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    #path= path.replace("cifar-10-batches-py","")
    #CIFAR is already adding cifar-10-batches-py/
    trainset = CIFAR10(path, train=True, download=True, transform=transform)
    testset = CIFAR10(path, train=False, download=True, transform=transform)
    #trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    #testloader = DataLoader(testset, batch_size=32)
    #num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return (trainset,None) , (testset,None) 



class FLModel(mlflow.pyfunc.PythonModel):

    def create_strategy(self, num_rounds: int):
        import flwr
        from flwr.server.strategy import FedAvg
        from flwr.server.client_proxy import ClientProxy
        from typing import List, Tuple, Union, Optional, Dict
        from flwr.common import FitRes, Scalar, Parameters, parameters_to_ndarrays
        import numpy as np
        class SaveModelStrategy(FedAvg):

            def aggregate_fit(
                    self,
                    server_round: int,
                    results: List[Tuple[ClientProxy, FitRes]],
                    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

                    # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                    aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

                    if aggregated_parameters is not None and server_round == num_rounds:
                        # Convert `Parameters` to `List[np.ndarray]`
                        aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

                        # Save aggregated_ndarrays
                        print(f"Saving round {server_round} aggregated_ndarrays...")
                        np.savez(f"results.npz", *aggregated_ndarrays)

                    return aggregated_parameters, aggregated_metrics

        return SaveModelStrategy(
            min_available_clients=1,
            min_fit_clients=1,
            min_evaluate_clients=1
        )
    def create_model(self):

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader


        from typing import List, Tuple, Union, Optional, Dict
        class Net(nn.Module):
            def __init__(self) -> None:
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = Net().to(DEVICE) 

        class PythonModelWrapper():
            def __init__(self, model):
              self.model = net
              
            def predict(self, context, model_input):

              return self.model.fit
            def fit(self, x_train,y_train, epochs=1, batch_size=1, steps_per_epoch=3):
              #self.set_parameters(parameters)
              x_train = DataLoader(x_train, batch_size=batch_size, shuffle=True)
              
              #num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
              criterion = torch.nn.CrossEntropyLoss()
              optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
              for _ in range(epochs):
                  for images, labels in x_train: #trainloader:
                      images, labels = images.to(DEVICE), labels.to(DEVICE)
                      optimizer.zero_grad()
                      loss = criterion(self.model(images), labels)
                      loss.backward()
                      optimizer.step()

            def get_weight(self):
              return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            def set_weight(self,parameters):
                  params_dict = zip(self.model.state_dict().keys(), parameters)
                  state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                  return self.model.load_state_dict(state_dict, strict=True)
            def evaluate(self,x_test,y_test) -> Tuple[float,float]:
              """Validate the network on the entire test set."""
              x_test = DataLoader(x_test, batch_size=1)
              criterion = torch.nn.CrossEntropyLoss()
              correct, total, loss = 0, 0, 0.0
              with torch.no_grad():
                  for data in x_test: #testloader
                      images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                      outputs = self.model(images)
                      loss += criterion(outputs, labels).item()
                      _, predicted = torch.max(outputs.data, 1)
                      total += labels.size(0)
                      correct += (predicted == labels).sum().item()
              accuracy = correct / total
              return loss, accuracy
            def load_data(self,path):
                  return load_data(path)
        return PythonModelWrapper(model=model)

model = FLModel()

import cloudpickle
with open('model_cifar_pytorch.pk', 'wb') as f:
    cloudpickle.dump(model, f)
import mlflow as mlf
#mlflow.set_tracking_uri("http://localhost:5002/")
#mlflow.set_experiment("test2")

#mlf.pyfunc.log_model(
#         python_model=model,
#         artifact_path="modello_pytorch_new",
#         registered_model_name="pickle_pytorch",
#     )


