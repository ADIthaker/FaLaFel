import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from collections import defaultdict
import copy
import numpy as np
import socket
import pickle
import asyncio
import hashlib
import time
import warnings
warnings.filterwarnings("ignore")

def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return x, y

x, y = sklearn_to_df(load_breast_cancer())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"sizes train = {len(y_train)} and test = {len(y_test)} ")
client_IPs = [('localhost', 8001), ('localhost', 8002), ('localhost', 8003), ('localhost', 8004), ('localhost', 8005)]

def hash_file(file_name):
    BUF_SIZE = 65536
    sha1 = hashlib.sha1()
    with open(file_name, 'rb') as f:
        print("Calculating Hash")
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

good_updates =[]
bad_updates=[]
class FederatedServerLogReg():

    def __init__(self, addr, client_IPs, min_loss, epochs, client_file):
        self.client_hash = hash_file(client_file)
        self.losses = []
        self.max_epochs = epochs
        self.min_loss = min_loss
        self.train_accuracies = []
        self.client_IPs = client_IPs
        self.IP = addr[0]
        self.n = len(self.client_IPs)
        self.PORT = addr[1]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.IP, self.PORT))
        self.rounds = 0
        self.curr_loss = 100
        self.latest_epoch = 0
        self.round_updates = defaultdict(list)
        self.start = [-1  for i in range(self.n)]
        self.end = [-1  for i in range(self.n)]
        self.x = self._transform_x(x)
        self.y = self._transform_y(y)
        self.weights = np.zeros(self.x.shape[1])
        self.bias = 0

    async def _recv_start(self):
        start_sent = True
        for i in self.start:
            if i != 1:
                start_sent = False
        while not start_sent:
            print("GETTING START MESSAGES ...")
            print(self.start)
            packet = self.sock.recvfrom(10000)
            ser_msg, _ = packet
            msg = pickle.loads(ser_msg)
            if msg["type"] == "start":
                print("GOT A START MESSAGE", msg['id'])
                self.start[msg["id"]-1] = 1
                start_sent = True
                for i in self.start:
                    if i != 1:
                        start_sent = False

    async def round(self):
        #poll all clients for updates
        print("ROUND START", self.rounds)
        msg = {
            "type": "gradient_request",
            "round": self.rounds,
            "global_weights": self.weights if self.rounds>0 else None
        }
        ser_msg = pickle.dumps(msg)
        for cl in self.client_IPs:
            self.sock.sendto(ser_msg, cl)
        await self._recv_gradients()

    async def _recv_gradients(self):
        while len(self.round_updates[self.rounds]) < self.n:
            # print("GETTING MESSAGES ...")
            packet = self.sock.recvfrom(10000)
            ser_msg, _ = packet
            msg = pickle.loads(ser_msg)
            print("GOT ONE FROM", msg["id"])
            if msg["type"] == "gradient":
                # print("GRADIENT", msg['gradient'])
                if msg["trust"] == self.client_hash:
                    self.round_updates[self.rounds].append(msg["gradient"])
                else:
                    print("HASH DID NOT MATCH, CANT TRUST CLIENT", msg['id'], "in round", self.rounds)
                    self.round_updates[self.rounds].append((0,0))
            elif msg["type"] == "end":
                self.end[msg['id']-1] = 1
        print("Gathered all gradients with len: ", len(self.round_updates[self.rounds]))
        await self.fit()
        
        return self.rounds
    async def send_global_updates(self):
        msg = {
            "type": "global_update",
            "global_weights": self.weights
        }
        ser_msg = pickle.dumps(msg)
        for cl in self.client_IPs:
            self.sock.sendto(ser_msg, cl)
    async def fit(self):

        while self.latest_epoch < self.max_epochs: #make these epochs count into condition for reaching a min loss value
            time.sleep(0.5) # sleep for 0.5seconds before asking for updates
            # error_w, error_b = self.compute_gradients(x, y, pred)

            #take average of all gradients recieved.
            ls_updates = self.round_updates[self.rounds]
            #take average of all 
            error_w, error_b = (0,0)
            for up in ls_updates:
                if not isinstance(up[0], int):
                    error_w, error_b = np.zeros(up[0].shape), np.zeros(up[1].shape)

            if isinstance(error_w, int):
                print("NO UPDATES RECVD")
                self.rounds += 1
                print("Start a new round", self.rounds)
                await self.round()
                return
            
            no_updates = 0
            for i, k in ls_updates:
                if not isinstance(i, int) and not isinstance(k, int):
                    error_w += i
                    error_b += k
                    no_updates += 1

            error_w = error_w / no_updates
            error_b = error_b / no_updates
            print("Average Gradient: ", error_w)

            self.update_model_parameters(error_w, error_b)

            x_dot_weights = np.matmul(self.weights, self.x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            self.curr_loss = loss
            self.latest_epoch += 1
            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            round_accuracy = accuracy_score(self.y, pred_to_class)
            self.train_accuracies.append(round_accuracy)
            self.losses.append(loss)

            if round_accuracy<0.4:
                bad_updates.append((error_w, error_b))
            elif round_accuracy>0.5:
                good_updates.append((error_w, error_b))
            await self.send_global_updates()
            print(self.losses[-1], self.train_accuracies[-1])
            print(self.weights)
            self.rounds += 1
            print("Start a new round", self.rounds)
            await self.round()
        print("Sending END TO CLIENTS")
        end_msg =  {
                    "type": "end",
                }
        for cl in self.client_IPs:
            ser_msg = pickle.dumps(end_msg)
            self.sock.sendto(ser_msg, cl)


    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(self.x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - 0.1 * error_w
        self.bias = self.bias - 0.1 * error_b

    def predict(self, x):
        print("Final weights: ", self.weights)
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)

if __name__ == "__main__":
    print(hash_file("client.py"))
    lr = FederatedServerLogReg(("localhost", 8000), client_IPs, 0.01, 40, "client.py")
    asyncio.run(lr._recv_start())
    print("GOT ALL START MESSAGES")
    start=time.time()
    asyncio.run(lr.round())
    print("time taken: ", time.time()-start)
    pred = lr.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print(accuracy)
    with open(f"log_{time.time()}.txt", 'w') as f:
        f.write(f"Training Accuracies = {lr.train_accuracies}\n")
        f.write(f"Losses = {lr.losses}")
    with open(f"updates_{time.time()}.txt", 'w') as f:
        f.write(f"Good updates = {good_updates}\n")
        f.write(f"Bad updates = {bad_updates}")


