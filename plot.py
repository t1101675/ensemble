import matplotlib.pyplot as plt

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

fig = plt.figure(figsize=(10, 10))
sub = fig.add_subplot(111)

X = [20, 40, 60, 80, 100, 120, 140]
Y005 = [0.8305, 0.8245, 0.8199, 0.8208, 0.8195, 0.8189, 0.8189]
Y01 = [0.8230, 0.8116, 0.8119, 0.8111, 0.8105, 0.8063, 0.8090]
Y02 = [0.8360, 0.8262, 0.8291, 0.8282, 0.8197, 0.8191, 0.8184]



l_005, = sub.plot(X, Y005, '^-', linewidth=3, ms=10)
l_01, = sub.plot(X, Y01, '^-', linewidth=3, ms=10)
l_02, = sub.plot(X, Y02, '^-', linewidth=3, ms=10)

# fig, ax = plt.subplots()
# ax.set_xscale("log")
# plt.semilogx(Y_acc)

plt.grid()
plt.tick_params(labelsize=15)

plt.xlabel("Training Times", font1)
plt.ylabel("RMSE", font1)

plt.legend(handles=[l_005, l_01, l_02], labels=['samp rate = 0.05', 'samp rate = 0.1', 'samp rate = 0.2'], prop=font1)
plt.savefig("images/bagging.png")
