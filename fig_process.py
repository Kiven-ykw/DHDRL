import json
import matplotlib.pyplot as plt
import numpy as np

def data_porcess():

    window = 500

    filename = 'RatevsTimeslot/rate500000R100AP6UE3QoS.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    plt.plot(r, label='rate100', marker="v", markevery=500000, markersize=6, markerfacecolor='none', linewidth=1)

    plt.xlabel('The number of time slots')
    plt.ylabel('SE (bps/Hz)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


data_porcess()
