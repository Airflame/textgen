from network import Network

net = Network()
net.load_data()
net.train(2000, 0.1)
print(net.sample(6))
print(net.sample(7))
print(net.sample(8))