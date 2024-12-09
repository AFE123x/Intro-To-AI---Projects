from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    """
    A simple perceptron model with a single weight vector for classification tasks.
    """

    def __init__(self, dimensions):
        """
        Initializes the perceptron model with a weight vector of the given dimensions.
        
        Args:
            dimensions: The size of the input features (number of features)
        """
        super(PerceptronModel, self).__init__()
        self.w = Parameter(ones(1, dimensions))  # Initialize weights

    def get_weights(self):
        """
        Returns the weight vector of the perceptron model.
        
        Returns:
            The weight parameter (w) of the model.
        """
        return self.w

    def run(self, x):
        """
        Computes the output of the perceptron by taking the dot product of 
        the input vector (x) and the weight vector.
        
        Args:
            x: The input tensor with shape (batch_size x dimensions).
        
        Returns:
            The result of the dot product of the weight and input.
        """
        return tensordot(self.w, x, dims=self.w.dim())

    def get_prediction(self, x):
        """
        Returns the predicted class label based on the sign of the output from the perceptron.
        
        Args:
            x: The input tensor with shape (batch_size x dimensions).
        
        Returns:
            1 if the output score is non-negative, -1 otherwise.
        """
        score = self.run(x)
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Trains the perceptron model using the given dataset.

        The model updates its weights by applying the perceptron learning rule, 
        adjusting the weights if the prediction is incorrect.

        Args:
            dataset: The dataset used for training. It should contain inputs and labels.
        """
        with no_grad():
            data = DataLoader(dataset, batch_size=1)  # Use batch size of 1 for online learning

            m = 1
            while m != 0:  # Continue until no mistakes are made
                m = 0
                for batch in data:
                    x, label = batch['x'], batch['label']

                    prediction = self.get_prediction(x)

                    if prediction != label.item():
                        self.w += x * label.item()  # Update the weights
                        m += 1  # Count the number of mistakes

class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers, with multiple hidden layers for function approximation.
    """
    def __init__(self):
        """
        Initializes the regression model with one input layer, several hidden layers, and one output layer.
        The model is large enough to approximate sin(x) on the interval [-2pi, 2pi].
        """
        super(RegressionModel, self).__init__()

        input_size = 1
        hidden_size = 500
        output_size = 1
        learning_rate = .003
        self.num_epoch = 5000
        self.num_hidden_layers = 3

        self.linear_in = Linear(input_size, hidden_size)
        self.linear_middle = Linear(hidden_size, hidden_size)
        self.linear_out = Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        Forward pass through the network. The model computes the output by applying 
        the layers sequentially, using ReLU activations for the hidden layers.
        
        Args:
            x: A tensor with shape (batch_size x 1), representing input values.
        
        Returns:
            The predicted y-values with shape (batch_size x 1).
        """
        out = self.linear_in(x)
        out = relu(out)

        for i in range(self.num_hidden_layers):
            out = self.linear_middle(out)
            out = relu(out)

        out = self.linear_out(out)
        return out

    def get_loss(self, x, y):
        """
        Computes the mean squared error (MSE) loss for a batch of examples.

        Args:
            x: A tensor with shape (batch_size x 1) containing the predicted y-values.
            y: A tensor with shape (batch_size x 1) containing the true y-values.
        
        Returns:
            A tensor containing the MSE loss value.
        """
        x = self.forward(x)
        return mse_loss(x, y)

    def train(self, dataset):
        """
        Trains the regression model using the provided dataset, adjusting the model weights 
        to minimize the MSE loss using the Adam optimizer.

        Args:
            dataset: A PyTorch dataset object containing data to train on.
        """
        batch_size = 1

        # Choose an appropriate batch size based on the dataset length
        for i in range(128, 1, -1):
            if len(dataset) % i == 0:
                batch_size = i
                break

        data = DataLoader(dataset, batch_size=batch_size)
        epoch_count = 0
        for epoch in range(self.num_epoch):
            epoch_loss = 0.0
            epoch_count += 1
            for batch in data:
                x, label = batch['x'], batch['label']

                loss = self.get_loss(x, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(data)

            if epoch_count % 10 == 0:
                print(f"Epoch [{epoch+1}/{epoch_count}], Loss: {avg_loss:.4f}")

            if avg_loss <= 0.001:
                print(f'Training stopped at epoch {epoch+1} with loss {avg_loss:.4f}')
                break

class DigitClassificationModel(Module):
    """
    A neural network model for digit classification. The model is built with two hidden layers
    and one output layer to predict the class of a given digit image.
    """
    def __init__(self):
        """
        Initializes the digit classification model with two hidden layers and one output layer.
        """
        super(DigitClassificationModel, self).__init__()

        input_size = 28 * 28  # Input size for 28x28 images
        hidden_size1 = 256
        hidden_size2 = 128
        output_size = 10  # Output size for 10 digit classes (0-9)
        learning_rate = 0.1

        self.linear1 = Linear(input_size, hidden_size1)
        self.linear2 = Linear(hidden_size1, hidden_size2)
        self.linear3 = Linear(hidden_size2, output_size)

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def run(self, x):
        """
        Forward pass through the network. The model applies two hidden layers with ReLU activations 
        and an output layer to produce the logits for classification.
        
        Args:
            x: A tensor of shape (batch_size x input_size), representing the input images.
        
        Returns:
            A tensor of shape (batch_size x output_size), representing the logits for each class.
        """
        x = relu(self.linear1(x))  # First hidden layer with ReLU
        x = relu(self.linear2(x))  # Second hidden layer with ReLU
        x = self.linear3(x)        # Output layer (logits)
        return x

    def get_loss(self, x, y):
        """
        Computes the cross-entropy loss for the model's output compared to the true labels.
        
        Args:
            x: A tensor of shape (batch_size x output_size), representing the logits.
            y: A tensor containing the true labels with shape (batch_size).
        
        Returns:
            A tensor containing the cross-entropy loss.
        """
        logits = self.run(x)
        return cross_entropy(logits, y)

    def train(self, dataset):
        """
        Trains the digit classification model using the provided dataset.

        Args:
            dataset: A PyTorch dataset object containing labeled image data to train on.
        """
        batch_size = 64  # Batch size for training
        epochs = 10      # Number of epochs for training
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in data_loader:
                x, labels = batch['x'], batch['label']

                loss = self.get_loss(x, labels)

                self.optimizer.zero_grad()  # Reset gradients
                loss.backward()             # Compute gradients
                self.optimizer.step()       # Update parameters

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(data_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
