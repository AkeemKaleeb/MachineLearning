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
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        # Initialize the weights
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        # Compute dot product of weights and input
        return tensordot(self.w, x, dims = ([1], [1]))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # Get result of run
        result = self.run(x)

        # Return 1 if non-negative, -1 otherwise
        return 1 if result >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            while True:                                     # Train until forced break
                all_correct = True                          # Assume all correct

                for batch in dataloader:                    # Loop through every batch from the data loader
                    x = batch['x']                          # Assign x and y based on batch
                    y = batch['label']
                    prediction = self.get_prediction(x)
                    if prediction != y:
                        all_correct = False
                        self.w += y * x
                if all_correct:
                    break






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
        self.hidden_size1 = 200     # Size of hidden layers for more complex patterns (100-500)
        self.hidden_size2 = 100     # Size of hidden layers for more complex patterns (100-500)

        self.fc1 = Linear(1, self.hidden_size1)                     # Layer 1
        self.fc2 = Linear(self.hidden_size1, self.hidden_size2)     # Layer 2
        self.fc3 = Linear(self.hidden_size2, 1)                     # Layer 3



    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = relu(self.fc1(x))       # First ReLu equation
        x = relu(self.fc2(x))       # Second ReLu equation
        x = self.fc3(x)             # Convergence and Prediction
        return x

    
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
        y_prediction = self.forward(x)
        return mse_loss(y_prediction, y)
  

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
        trials = 1000       # Time to learn
        lr = 0.00025        # Convergence Rate (0.0001-0.01)
        batch_size = 16     # Amount of data sent at once (1-128)

        dataloader = DataLoader(dataset, batch_size, shuffle=True)      # Data Loader for training
        optimizer = optim.Adam(self.parameters(), lr)                   # Optimizer for convergence
        for trial in range(trials):                                     # Loop through custom number of trials
            total_loss = 0                                              # Initialize loss                                 
            for batch in dataloader:                                    # Loop through every batch
                x = batch['x']                                          # Assign x and y based on batch               
                y = batch['label']
                optimizer.zero_grad()                                   # Zero out gradients
                loss = self.get_loss(x, y)                              # Get loss
                loss.backward()                                         # Backward propagation
                optimizer.step()                                        # Move to next part of the optimizer
                total_loss += loss.item()                               # Add to total loss
            avg_loss = total_loss / len(dataloader)                     # Average loss
            print(f'Trial {trial+1}/{trials}, Loss: {avg_loss:.4f}')    # Print loss
            if avg_loss <= 0.02:                                        # Break if loss is low enough
                break

            







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
        hidden_size1 = 256          # Size of the first hidden layer
        hidden_size2 = 128          # Size of the second hidden layer

        self.fc1 = Linear(input_size, hidden_size1)         # First hidden layer
        self.fc2 = Linear(hidden_size1, hidden_size2)       # Second hidden layer
        self.fc3 = Linear(hidden_size2, output_size)        # Output layer



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
        x = relu(self.fc1(x))               # Apply ReLU activation after the first layer
        x = relu(self.fc2(x))               # Apply ReLU activation after the second layer
        x = self.fc3(x)                     # Output layer (no activation function)
        return x


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
        y_prediction = self.run(x)
        return cross_entropy(y_prediction, y)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """

        epochs = 5                      # Number of epochs
        learning_rate = 0.001           # Learning rate
        batch_size = 32                 # Batch size

        dataloader = DataLoader(dataset, batch_size, shuffle = True)
        optimizer = optim.Adam(self.parameters(), learning_rate)            # Optimizer for convergence
        for epoch in range(epochs):                                         # Loop through custom number of trials
            total_loss = 0                                                  # Initialize loss                                 
            for batch in dataloader:                                        # Loop through every batch
                x = batch['x']                                              # Assign x and y based on batch               
                y = batch['label']
                optimizer.zero_grad()                                       # Zero out gradients
                loss = self.get_loss(x, y)                                  # Get loss
                loss.backward()                                             # Backward propagation
                optimizer.step()                                            # Move to next part of the optimizer
                total_loss += loss.item()                                   # Add to total loss
            avg_loss = total_loss / len(dataloader)                         # Average loss
            validation_accuracy = dataset.get_validation_accuracy()         # Get validation accuracy

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}')       # Print loss
            if validation_accuracy >= .98:                                  # Break if loss is low enough
                break
