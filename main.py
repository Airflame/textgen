from network import Network

net = Network()
net.load_data()
net.train(2000, 0.1)
net.train(1000, 0.01)
for _ in range(100):
    print(net.sample())