from network import Network

net = Network()
net.load_data()
net.train(400, 0.1)
for _ in range(100):
    print(net.sample())