import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler as STD
import copy
import socket
import sys
import pickle
import time
import hashlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return x, y

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

class FederatedClientLogReg():

    def __init__(self, addr, server_addr, id):
        self.losses = []
        self.train_accuracies = []
        self.IP = addr[0]
        self.PORT = addr[1]
        self.id = id
        self.server_addr = server_addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.IP, self.PORT))
        #self.sock.settimeout(0.5)
        self.round_to_epoch = {}
        self.epoch_gradient = {}
        self.latest_gradient = []
        self.latest_epoch = 0
        self.start_sent = False
        self.end_training = False
    
    

    def round_send(self):
        while True:
            packet = self.sock.recvfrom(10000)
            ser_msg, _ = packet
            msg = pickle.loads(ser_msg)
            print("in round_send, got a messsage")
            trust = hash_file(__file__)
            if msg["type"] == "gradient_request":
                #send my reply
                reply = {
                    "type": "gradient",
                    "gradient": self.latest_gradient,
                    "id": self.id,
                    "trust": trust,
                }
                ser_reply = pickle.dumps(reply)
                self.sock.sendto(ser_reply, self.server_addr)
                self.round_to_epoch[msg["round"]] = self.latest_epoch
                break
            elif msg["type"]=="end":
                self.end_training = True
                break

    def round_global_weights(self):
        while True:
            packet = self.sock.recvfrom(10000)
            ser_msg, _ = packet
            msg = pickle.loads(ser_msg)
            print("received global updates")
            if msg["type"] == "global_update":
                self.weights = msg["global_weights"]
                break
            elif msg["type"]=="end":
                self.end_training = True
                break



    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.random.uniform(low=0, high=1, size=x.shape[1])
        self.bias = 0.1
        i = 0
        # while not self.end_training:
        for i in range(150):
            # time.sleep(1) # collecting data
            print("Starting New Epoch", i)

            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            print("local loss = ", loss)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.latest_gradient = (error_w, error_b)
            self.epoch_gradient[i] = self.latest_gradient
            self.latest_epoch = i
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

        #     if i == 3:
        #         # send startup message
        #         msg = {
        #             "type": "start",
        #             "id": self.id
        #         }
        #         ser_msg = pickle.dumps(msg)
        #         self.start_sent = True
        #         print("SENDING START TO SERVER")
        #         self.sock.sendto(ser_msg, self.server_addr)
                
        #     if self.start_sent:
        #         print("Waiting for a request from server and sending my reply")
        #         self.round_send()
        #         # self.round_global_weights()
        #     i+=1
        # msg =  {
        #             "type": "end",
        #             "id": self.id
        #         }
        # print("Sending END TO SERVER")
        # ser_msg = pickle.dumps(msg)
        # self.sock.sendto(ser_msg, self.server_addr)
        # self.sock.recvfrom(10000)
            print("ACC", accuracy_score(y, pred_to_class), f1_score(y, pred_to_class))
            print("PREC-REC", precision_score(y, pred_to_class), recall_score(y, pred_to_class))
        
        plt.plot(self.losses)
        plt.plot(np.multiply(self.train_accuracies,10))
        plt.show()
        print(accuracy_score(y, pred_to_class))

        
    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b, lr=0.00018, beta=0.01):
        self.weights = self.weights - lr * (error_w ) #+ 2*beta*self.weights
        self.bias = self.bias - lr * (error_b) #+ 2*beta*self.bias

    def predict(self, x):
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
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)

if __name__ == "__main__":
    args = sys.argv
    id = int(args[1])

    x, y = sklearn_to_df(load_breast_cancer())
    x, y = shuffle(x, y, random_state=0)
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    no_rows = len(x) // 5
    x = x.iloc[no_rows*(id-1):no_rows*id, :] #split dataset at client
    y = y.iloc[no_rows*(id-1):no_rows*id]
    print(type(x))
    # normalize data
    scaler = STD()
    x = scaler.fit_transform(x)
    #x = x.values
    print(type(x))
    print("split dataset")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    lr = FederatedClientLogReg(('localhost', 8000+id), ('localhost', 8000), id)
    print("Starting Fit")
    print("length of dataset = ",len(x_train), len(x_test))
    lr.fit(x_train, y_train, epochs=20)
    # print(lr.train_accuracies[-1])

# lr.fit(x_train, y_train, epochs=150)
# pred = lr.predict(x_test)
# accuracy = accuracy_score(y_test, pred)
# print(accuracy)

