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
    def __init__(self, dimensions):
        #print("\n\nstart init")
        super(PerceptronModel, self).__init__()
        # Initialize the weights as a PyTorch Parameter
        #print(f"dims:{dimensions}")
        self.w = Parameter(ones(1, dimensions))
        #print(f"Initialized weights shape: {self.w.shape}")  # Should print torch.Size([1, 2])
        #print(f"self.w.dim() = {self.w.dim()}")
        #print(f"Initialized weights:\n{self.w}")  # Should print torch.Size([1, 2])

    def get_weights(self):
        return self.w

    def run(self, x):
        # Compute the score using tensordot       
        #print("\nrun") 
        ##print(f"weights:\n{self.w}") 
        ##print(f"x:\n{x}") 
        #print(f"td:\n{tensordot(self.w, x, dims=self.w.dim())}\n")
        return tensordot(self.w, x, dims=self.w.dim())

    def get_prediction(self, x):
        # Return 1 if the score is positive, otherwise -1
        score = self.run(x)
        #print("\nGet prediction")
        ##print(f"x:{x}")
        #print(f"score:{score}")
        return 1 if score >= 0 else -1

    def train(self, dataset):
        #print("train")
        with no_grad():
            data = DataLoader(dataset, batch_size=1)

            m = 1
            while m != 0:
                m = 0
                for batch in data:
                    x, label = batch['x'], batch['label']
                    #print(f"(x,label) ({x},{label})\n")
                    prediction = self.get_prediction(x)
                    #print(f"item:{label.item()}")
                    # if the prediction doesnt equal the label, update the weight vector
                    if prediction != label.item():
                        self.w += x * label.item()
                        m += 1
                #print(f"miss classifications={m}")
            #print(f"loop over")



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super(RegressionModel, self).__init__()

        input_size = 1
        hidden_size = 500
        output_size = 1
        learning_rate = .003
        self.num_epoch = 5000
        self.num_hidden_layers = 3

        # LR 1000 epoch
        # .002 = 0.000288
        # .003 = 0.000113
        # .004 = 0.000127
        # .005 = 0.000120
        # .006 = 0.000327
        # .007 = 0.000095
        # .01 =  0.000493

        # LR 5000 epoch
        # .002 = 0.
        # .003 = 0.000024
        # .004 = 0.
        # .005 = 0.
        # .006 = 0.
        # .007 = 0.
        # .01 =  0.
        



        # input to the hidden layer
        self.linear_in = Linear(input_size, hidden_size)

        # can put more layers in here
        # hidden layer to the output
        self.linear_middle = Linear(hidden_size, hidden_size)

        self.linear_out = Linear(hidden_size, output_size)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)



    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #print("\nforward")
        out = self.linear_in(x)
        #print(f"in:\n{out}")
        out = relu(out)
        #print(f"relu:\n{out}")

        # added loop for hidden layers
        for i in range(self.num_hidden_layers):
            out = self.linear_middle(out)
            #print(f"middle_{i}:\n{out}")
            out = relu(out)
            #print(f"relu_{i}:\n{out}")
        
        out = self.linear_out(out)
        #print(f"out:\n{out}")
        return out

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
       # print()
        #print(mse_loss(x,y))
        x = self.forward(x)
        return mse_loss(x,y)

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        #print("\n\ntrain")
        batch_size = 1
        
        # finds a batch size that is between 128 and 1 and works with the data set size
        for i in range(128, 1, -1):
            if len(dataset) % i == 0:
                batch_size = i
                break
        print(f"DS Length:{len(dataset)}")
        print(f"Batch Size:{batch_size}")

        data = DataLoader(dataset, batch_size=batch_size)
        epoch_count = 0
        for epoch in range(self.num_epoch):
            epoch_loss = 0.0
            epoch_count += 1
            for batch in data:
                x, label = batch['x'], batch['label']
                #print(f"x:\n{x}\nlabel:\n{label}\n")
                
                # get predicted y values & calculate loss tensor
                loss = self.get_loss(x, label)

                # reset gradients
                self.optimizer.zero_grad()
                # calculate gradient
                loss.backward()
                # update weights
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(data)


            if epoch_count % 10 == 0:
                print(f"Epoch [{epoch+1}/{epoch_count}], Loss: {avg_loss:.4f}")

            # stopping if loss is below the threshold
            # 0.02 or better for full points
            if avg_loss <= 0.001:
               print(f'Training stopped at epoch {epoch+1} with loss {avg_loss:.4f}')
               break


class DigitClassificationModel(Module):
    def __init__(self):
        """
        Initialize the model with an input layer, two hidden layers, and an output layer.
        """
        super(DigitClassificationModel, self).__init__()
        input_size = 28 * 28 
        hidden_size1 = 256
        hidden_size2 = 128
        output_size = 10
        learning_rate = 0.01

        # Define layers
        self.linear1 = Linear(input_size, hidden_size1)
        self.linear2 = Linear(hidden_size1, hidden_size2)
        self.linear3 = Linear(hidden_size2, output_size)

        # Define optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def run(self, x):
        """
        Forward pass through the network.
        """
        x = relu(self.linear1(x))  # First hidden layer with ReLU
        x = relu(self.linear2(x))  # Second hidden layer with ReLU
        x = self.linear3(x)        # Output layer (logits)
        return x

    def get_loss(self, x, y):
        """
        Compute the loss for a batch of examples.
        """
        logits = self.run(x)
        return cross_entropy(logits, y)

    def train(self, dataset):
        """
        Train the model using the dataset.
        """
        batch_size = 64  # Batch size for training
        epochs = 10      # Number of epochs for training
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in data_loader:
                x, labels = batch['x'], batch['label']

                # Perform forward pass and compute loss
                loss = self.get_loss(x, labels)

                # Backpropagation
                self.optimizer.zero_grad()  # Reset gradients
                loss.backward()             # Compute gradients
                self.optimizer.step()       # Update parameters

                epoch_loss += loss.item()

            # Print average loss for the epoch
            avg_loss = epoch_loss / len(data_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
