# GradientDescentOptimizer from scratch !!!!

I have made a very simple implementation of gradient descent from scratch using Python,numpy,pandas.
I have only included 1 loss function for now which is mean squared error. Later i might add more.

After running the optimier, use self.w, self.b, self.errors, self.lr_decay to get weights, biases, errors, learning rate decay respectively. Here self is the name of the GradientDescentOptimizer class you have defined.

ihave also attached 3 notebooks where i have used my gradient descent algorithm of datasets i have created. So check those out.

It supports the following features:
- **Mean Squared Error (MSE)** loss function
- **Exponential learning rate decay** (optional)
- **Early stopping** based on convergence criteria (optional)
- Tracking of errors and learning rate history

## Features
- Adjustable **epochs**, **learning rate**, and **tolerance**
- Option for **variable learning rate** (exponential decay from `lr` to `lr_f`)
- **Early stopping** to avoid overtraining
- Returns convergence message at the end of training
## Parameters

| Parameter       | Description |
|-----------------|-------------|
| `epochs`        | Maximum number of iterations (default: 1000) |
| `lr`            | Initial learning rate (default: 0.01) |
| `lr_f`          | Final learning rate for exponential decay (default: 0.0001) |
| `loss`          | Loss function (only `"mse"` supported) |
| `random_state`  | Seed for reproducibility (default: 42) |
| `tolerance`     | Convergence threshold (default: 0.0000001) |
| `variable_lr`   | Enable exponential learning rate decay (default: False) |
| `early_stopping`| Stop early on convergence (default: False) |

Author: **Ayush Nandi**
