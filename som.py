import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SOM(nn.Module):
    def __init__(self, m, n, dim, epochs, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.grid_size = m * n
        self.m, self.n = m, n
        self.dim = dim
        self.epochs = epochs

        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        # torch.manual_seed(0)
        self.weights = torch.randn(m * n, dim)
        self.locations = torch.Tensor(self.neuron_locations()).long()
        self.pairwise_dist = nn.PairwiseDistance(p=2)

    def neuron_locations(self):
        return [[i, j] for i in range(self.m) for j in range(self.n)]

    def map_vecs(self, input_vecs):
        to_return = []
        for vec in input_vecs:
            diff = torch.norm(vec - self.weights, dim=1)
            _, min_index = torch.min(diff, 0)
            to_return.append(self.locations[min_index])

        return to_return

    def forward(self, x, it):
        dists = self.pairwise_dist(x, self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index]

        learning_rate_op = 1.0 - it/self.epochs
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        location_dist = torch.sum(torch.pow(self.locations - bmu_loc, 2), 1)
        # RBF kernel for similarity
        neighbourhood_func = torch.exp(-(torch.div(location_dist.float(), 2 * sigma_op ** 2)))

        learning_rate_op = alpha_op * neighbourhood_func
        learning_rate_multiplier = learning_rate_op.unsqueeze(1).repeat(1, self.dim)

        delta = torch.mul(learning_rate_multiplier, x - self.weights)
        new_weights = torch.add(self.weights, delta)

        self.weights = new_weights


if __name__ == "__main__":

    # config
    m = 70
    n = 90
    epochs = 500

    #Training inputs for RGBcolors
    colors = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])

    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']


    data = torch.Tensor(colors).float()

    #Train a 20x30 SOM with 100 iterations
    som = SOM(m, n, 3, epochs)
    for epoch in range(epochs):
        for i in range(len(data)):
            som(data[i], epoch)

    #Get output grid
    image_grid = som.weights.view(m, n, -1).data.numpy()

    #Map colours to their closest neurons
    mapped = som.map_vecs(data)

    #Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    print("len map: ", len(mapped))
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], color_names[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()