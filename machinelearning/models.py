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
        self.linear1 = Linear(input_size, hidden_size)

        # can put more layers in here
        # hidden layer to the output
        self.linear2 = Linear(hidden_size, output_size)

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
        s1 = self.linear1(x)
        #print(f"s1:\n{s1}")
        s2 = relu(s1)
        #print(f"s2:\n{s2}")
        s3 = self.linear2(s2)
        #print(f"s3:\n{s3}")
        return s3

    
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
        print("\n\ntrain")
        #self.train()
        batch_size = 1
        
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


            if epoch_count % 100 == 0:
                print(f"Epoch [{epoch+1}/{epoch_count}], Loss: {avg_loss:.4f}")

            # Early stopping if loss is below the threshold
            #if avg_loss <= 0.002:
            #    print(f'Training stopped at epoch {epoch+1} with loss {avg_loss:.4f}')
            #    break


            







class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
