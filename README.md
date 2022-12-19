# Introduction_to_GAN

link to colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NEWKNP/Introduction_to_GAN/blob/main/Introduce_to_GAN.ipynb)

# Learning path
![Directory Structure](/asset/flow.png "From zero to GAN's junior user")

# Set up
clone material form my Github
```.bash
!git clone https://github.com/NEWKNP/Introduction_to_GAN.git
```
ตรวจสอบว่า clone สำเร็จ
```.bash
!ls
#-> Introduction_to_GAN  sample_data
```

# Machine learing and deep learning

## Percepton
a simple neuron network model

### vector base
>NumPy is a Python library used for working with arrays.

It also has functions for working in domain of linear algebra, and matrices.
>In Python we have lists that serve the purpose of arrays, but NumPy aims to provide an array object that is up to 50x faster than traditional Python lists.  
    Doc: [Here](https://numpy.org/doc/stable/user/quickstart.html)  
    or use help()
 ```.bash
import numpy as np
```
```.bash
# OR problem
X = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
Y = [1, 1, 1, -1]
X = np.hstack((np.array(X), np.ones((len(X), 1)))) # add constant variables column (B)
X
'''
-> array([[ 1.,  1.,  1.],
          [ 1., -1.,  1.],
          [-1.,  1.,  1.],
          [-1., -1.,  1.]])
'''
```
```.bash
ini_w = X[np.random.randint(0, len(X))].copy() #numpy anounment
ini_w
#-> array([-1.,  1.,  1.])
```
```.bash
np.dot(X[0], ini_w) # 1*(-1) + 1*(-1) + 1*1
#-> 1.0
```
```.bash
Y[0] * np.dot(X[0], ini_w) # 1*3
#-> 1.0
```
```.bash
print(np.dot(X[3], ini_w)) # (-1)*1 + (-1)*1 + 1*1
print(Y[3] * np.dot(X[3], ini_w)) # (-1)*(-1)
#-> 1.0
#-> -1.0
```
```.bash
def perceptron(X, Y, lr=0.1, w=None):
  """
  X = input data
  Y = label
  lr = learning rate
  w = weight or training variables or decision boundary
  Return: decision boundary
  """
  # Homogeneous coordinates
  X = np.hstack((np.array(X), np.ones((len(X), 1)))) # add constant variables column (B)
  if w is None:
    w = X[np.random.randint(0, len(X))].copy() # initial weight
  done = False
  while not done:
    done = True
    for i, x in enumerate(X):      # calculate each data point
      if Y[i] * np.dot(x, w) <= 0: # decision boundary still wrong classify
        w += lr * x * Y[i]         # update weight
        done = False               # set unfinish
  return w
```
```.bash
w = perceptron(X, Y, lr=1e-1)
print(w)
#-> [1.3 0.3 0.7 0.7]
```
```.bash
from matplotlib import pyplot as plt #library for plot
def plot_hyperplane2d(X, Y, w):
  # just plot
  X = np.array(X)
  Y = np.array(Y)
  plt.plot(X[Y==1, 0], X[Y==1, 1], 'go')   # green is 1
  plt.plot(X[Y==-1, 0], X[Y==-1, 1], 'ro') # red is -1
  xlim = plt.gca().get_xlim()
  slope = -w[0] / w[1]
  bias = -w[-1] / w[1]
  plt.plot(xlim, [xlim[0] * slope + bias, xlim[1] * slope + bias], 'b')
```
```.bash
plot_hyperplane2d(X, Y, w)
```
### gradient base
![Directory Structure](/asset/dnn_framework.png "Framework")
```.bash
import torch
from torch import nn, Tensor, from_numpy, optim
import torch.nn.functional as F
```
```.bash
# storage
lst = [1,2]
print("List: ", lst)
print("Numpy: ", np.array(lst))
print("Tensor: ", Tensor(lst))
#-> List:  [1, 2]
#-> Numpy:  [1 2]
#-> Tensor:  tensor([1., 2.])
```
```.bash
#do hstack, initial weight, and dot vector
nn.Linear(1,1)
#-> Linear(in_features=1, out_features=1, bias=True)
```  
```.bash
input_size = 2 # length of an input
outpur_size = 1 
perceptron_torch = nn.Sequential(
                    nn.Linear(input_size,outpur_size),
                    nn.Tanh()
                  )
perceptron_torch
'''
-> Sequential(
     (0): Linear(in_features=2, out_features=1, bias=True)
     (1): Tanh()
   )
'''
```  
```.bash
# OR problem
X = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
Y = [1, 1, 1, -1]     
```
```.bash
X_tensor = Tensor(X)
Y_pred = perceptron_torch(X_tensor) # feed forward
print(Y_pred)
print(torch.round(Y_pred))
'''
-> tensor([[-0.5041],
           [ 0.6341],
           [-0.8889],
           [-0.1131]], grad_fn=)
   tensor([[-1.],
           [ 1.],
           [-1.],
           [-0.]], grad_fn=)
'''
``` 
### learning
how math model become machine learning

Losss function
>TD;LR: Basline
>regression problem: Mean Square Error
>classification problem: Cross Entropy

You can know about loss on [this](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23)  
loss function in pytorch: click [here](https://pytorch.org/docs/stable/nn.html#loss-functions)  

```.bash
def get_loss(Y_pred, Y_test, loss=nn.MSELoss()):
  return loss(Y_pred, Y_test)
```
```.bash
Y_tensor = Tensor(Y).reshape(4,1)
loss = get_loss(Y_pred, Y_tensor, loss=nn.MSELoss())
loss.item()
#-> 1.6876214742660522 
```

You can know more about learning optimization on [here](https://github.com/jettify/pytorch-optimizer)  
For mathematics click [here](https://www.youtube.com/watch?v=uJryes5Vk1o&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=10)  
```.bash
# learning algorithm
optimizer = optim.SGD(perceptron_torch.parameters(), lr=1e-2)
```
```.bash
# training
# Run the training loop
for epoch in range(50000): # 50000 epochs at maximum    
  # Zero the gradients
  optimizer.zero_grad()
      
  # Perform forward pass
  Y_pred = perceptron_torch(X_tensor)
      
  # Compute loss
  loss = get_loss(Y_pred, Y_tensor, loss=nn.MSELoss())
      
  # Perform backward pass
  loss.backward()
      
  # Perform optimization
  optimizer.step()
  
  # Print epoch
  if epoch % 5000 == 0:
    print(f'Epoch {epoch+1} loss: {loss.item()}')
'''
-> Epoch 1 loss: 1.6876214742660522
   Epoch 5001 loss: 0.004373827017843723
   Epoch 10001 loss: 0.002088001696392894
   Epoch 15001 loss: 0.0013639380922541022
   Epoch 20001 loss: 0.0010106898844242096
   Epoch 25001 loss: 0.0008019705419428647
   Epoch 30001 loss: 0.0006642726366408169
   Epoch 35001 loss: 0.0005667562363669276
   Epoch 40001 loss: 0.0004940598737448454
   Epoch 45001 loss: 0.00043782772263512015 
'''
```
```.bash
print(torch.round(Y_pred))
print(Y_tensor) 
'''
-> tensor([[ 1.],
           [ 1.],
           [ 1.],
           [-1.]], grad_fn=)
   tensor([[ 1.],
           [ 1.],
           [ 1.],
           [-1.]])
'''
```
>Note activation function and learning algorithm will discuss later  
### implement as object
```.bash
class Perceptron(nn.Module):
  def __init__(self, input_dim, output_dim):
    """
    Params
      input_dim: length of a input data
      output_dim: length of a output (Must be same as Y)
    Define
      fc: neuron layer
      act_func: activation function 
      loss: loss function 
      optimizer: gradient optimizer 
    """
    super(Perceptron, self).__init__()          # inherit class
    self.fc = nn.Linear(input_dim, output_dim)
    self.act_func = nn.Tanh()                   # TanH activation function
    self.criterion = nn.MSELoss()                   # Mean Square error
    self.optimizer = optim.SGD(self.parameters()
                                     , lr=1e-2) # Stochastic gradient descent

  def forward(self, x):
    """
    feed forward the data through each neuron layer
    x: data, type tensor, size 2x1
    """
    x = self.fc(x)
    x = self.act_func(x)
    return x

  def fit(self, X, Y, epochs=500):
    """
    Training loop
    """
    X_tensor = Tensor(X)
    Y_tensor = Tensor(Y).reshape(4,1)
    loss_record = []
    for epoch in range(epochs):
      # Zero the gradients
      self.optimizer.zero_grad()
          
      # Perform forward pass
      Y_pred = self.forward(X_tensor)
          
      # Compute loss
      loss = self.criterion(Y_pred, Y_tensor)
      loss_record.append(loss.item()) #recording the loss
          
      # Perform backward pass
      loss.backward()
          
      # Perform optimization
      self.optimizer.step()
      
      # Print epoch
      if epoch % int(epochs/10) == 0:
        print(f'Epoch {epoch+1} loss: {loss.item()}')

    return loss_record
```
```.bash
or_perceptron = Perceptron(len(X[0]), 1)
history = or_perceptron.fit(X, Y, 10000)
'''
-> Epoch 1 loss: 1.6121445894241333
   Epoch 1001 loss: 0.02551443502306938
   Epoch 2001 loss: 0.011782277375459671
   Epoch 3001 loss: 0.00755075691267848
   Epoch 4001 loss: 0.005526163149625063
   Epoch 5001 loss: 0.004346323199570179
   Epoch 6001 loss: 0.003576214425265789
   Epoch 7001 loss: 0.003034995635971427
   Epoch 8001 loss: 0.0026343264617025852
   Epoch 9001 loss: 0.0023260070011019707
'''
```
```.bash
Y_pred = or_perceptron.forward(X_tensor)
print(Y_pred)
print(torch.round(Y_pred))
print(Y_tensor)
'''
-> tensor([[ 1.0000],
           [ 0.9473],
           [ 0.9473],
           [-0.9473]], grad_fn=)
   tensor([[ 1.],
           [ 1.],
           [ 1.],
           [-1.]], grad_fn=)
   tensor([[ 1.],
           [ 1.],
           [ 1.],
           [-1.]])
'''
```
## AutoEncoder
```.bash
X = np.eye(8).astype(np.float32) # identity matrix
X
'''
-> array([[1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1.]], dtype=float32)
'''
```
```.bash
class Autoencoder(nn.Module):
  def __init__(self, input_dim, output_dim, lr=3.5):
    """
    Params
      input_dim: length of a input data
      output_dim: length of a output (Must be same as Y)
    Define
      fc: neuron layer
      act_func: activation function 
      loss: loss function 
      optimizer: gradient optimizer 
    """
    super(Autoencoder, self).__init__()
    middle_dim = int((input_dim + output_dim)/2)
    self.encoder1 = nn.Linear(input_dim, middle_dim)
    self.encoder2 = nn.Linear(middle_dim, output_dim)
    self.decoder1 = nn.Linear(output_dim, middle_dim)
    self.decoder2 = nn.Linear(middle_dim, input_dim)
    self.act_func = nn.ReLU()
    self.decision = nn.Sigmoid()
    self.loss = nn.BCELoss()
    self.optimizer = optim.SGD(self.parameters(), lr=lr)

  def forward(self, x):
    """
    feed forward the data through each neuron layers
    x: data, type tensor
    """
    x = self.encode(x)
    x = self.decode(x)
    return x

  def encode(self, x):
    """
    encoding input data
    """
    x = self.encoder1(x)
    x = self.act_func(x)
    x = self.encoder2(x)
    return self.decision(x)

  def decode(self, x):
    """
    decoding input data
    """
    x = self.decoder1(x)
    x = self.act_func(x)
    x = self.decoder2(x)
    return self.decision(x)

  def get_loss(self, Y_pred, Y_test):
    """
    compute loss between our predict and label
    """
    return self.loss(Y_pred, Y_test)

  def fit(self, X, Y, epochs=500):
    """
    Training loop
    """
    X_tensor = from_numpy(X)
    Y_tensor = from_numpy(Y)
    loss_record = []
    for epoch in range(epochs):
      # Zero the gradients
      self.optimizer.zero_grad()
          
      # Perform forward pass
      Y_pred = self.forward(X_tensor)
          
      # Compute loss
      loss = self.get_loss(Y_pred, Y_tensor)
      loss_record.append(loss.item()) #recording the loss
          
      # Perform backward pass
      loss.backward()
          
      # Perform optimization
      self.optimizer.step()
      
      # Print epoch
      if epoch % int(epochs/10) == 0:
        print(f'Epoch {epoch+1} loss: {loss.item()}')

    return loss_record
```
# Vanilla GAN
## constuct vanilla GAN
![Directory Structure](/asset/5_compute_primitives_and_memory_layouts.png "Compute primitives and_memory layouts")
```.bash
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # training mode 
```
```.bash
class Generator(nn.Module):
  def __init__(self):
    """
    Similar decoder
    """
    super(Generator, self).__init__()
    self.n_features = 128
    self.n_out = 784
    self.fc0 = nn.Sequential(
                nn.Linear(self.n_features, 256), 
                nn.LeakyReLU(0.2)
                )
    self.fc1 = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2)
                )
    self.fc2 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2)
                )
    self.fc3 = nn.Sequential(
                nn.Linear(1024, self.n_out),
                nn.Tanh()
                )
  def forward(self, x):
    """
    feed forward the data through each neuron layers
    x: data, type tensor
    """
    x = self.fc0(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = x.view(-1, 1, 28, 28) # reshape
    return x

class Discriminator(nn.Module):
  def __init__(self):
    """
    Similar Encoder
    """
    super(Discriminator, self).__init__()
    self.n_in = 784
    self.n_out = 1
    self.fc0 = nn.Sequential(
                nn.Linear(self.n_in, 1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
    self.fc1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
    self.fc2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
    self.fc3 = nn.Sequential(
                nn.Linear(256, self.n_out),
                nn.Sigmoid()
                )
  def forward(self, x):
    """
    feed forward the data through each neuron layers
    x: data, type tensor
    """
    x = x.view(-1, 784) # Flatten
    x = self.fc0(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
```
```.bash
from time import sleep
from tqdm import tqdm
from torch.autograd.variable import Variable
from torchvision.utils import make_grid

class VanillaGAN(nn.Module):
  def __init__(self):
    """
    Similar autoencoder
    But
      encoder -> discriminator
      decoder -> generator
    """
    super(VanillaGAN, self).__init__()
    self.generator = Generator().to(device)
    self.discriminator = Discriminator().to(device)
    self.g_optim = optim.Adam(self.generator.parameters(), lr=2e-4)
    self.d_optim = optim.Adam(self.discriminator.parameters(), lr=2e-4)
    self.criterion = nn.BCELoss() # Binary Cross Entropy loss function

  def noise(self, size, n_features=128):
    return Variable(torch.randn(size, n_features)).to(device)

  def make_ones(self, size):
    return Variable(torch.ones(size, 1)).to(device)

  def make_zeros(self, size):
    return Variable(torch.zeros(size, 1)).to(device)

  def generate(self, n=25):
    return self.generator(self.noise(n))

  def train_discriminator(self, real_data, fake_data):
    # get vector size
    n = real_data.size(0)

    # prevent accumulate loss after through an epoch
    self.d_optim.zero_grad()
    
    # Step 2.1
    prediction_real = self.discriminator(real_data)
    error_real = self.criterion(prediction_real, self.make_ones(n))
    error_real.backward()

    # Step 2.2
    prediction_fake = self.discriminator(fake_data)
    error_fake = self.criterion(prediction_fake, self.make_zeros(n))
    error_fake.backward()

    # Perform optimization
    self.d_optim.step()

    # Step 2.3
    return error_real + error_fake

  def train_generator(self, fake_data):
    # get vector size
    n = fake_data.size(0)

    # prevent accumulate loss after through an epoch
    self.g_optim.zero_grad()
      
    # Step 3
    prediction = self.discriminator(fake_data)
    error = self.criterion(prediction, self.make_ones(n))
    
    # Perform backward pass
    error.backward()
    # Perform optimization
    self.g_optim.step()

    return error 

  def training_loop(self, trainloader, num_epochs=100, k=1):
    # for generate output images
    test_noise = self.noise(64)
    generate_images = []

    # histories
    g_losses = []
    d_losses = []

    # set to training mode
    self.generator.train()
    self.discriminator.train()

    # iterate cover dataset
    for epoch in range(num_epochs):
      with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        # reset loss
        g_error = 0.0   
        d_error = 0.0

        # iterate each batch
        for i, data in enumerate(tepoch):
          tepoch.update(1)
          sleep(0.5)
          imgs, _ = data
          n = len(imgs)
          real_data = imgs.to(device)
          # train discriminator in k steps (with the same real data)
          for j in range(k):
            # generate latent (Step 1.)
            fake_data = self.generator(self.noise(n)).detach()
            # train discriminator (Step 2.)
            d_error += self.train_discriminator(real_data, fake_data)
          # generate another latent
          fake_data = self.generator(self.noise(n))
          # train generator (Step 3.)
          g_error += self.train_generator(fake_data)
          tepoch.set_postfix({'Batch': i+1, 'G loss (in progress)': g_error.item()/(i+1),
                              'D loss (in progress)': d_error.item()/(i+1)})

        img = self.generator(test_noise).cpu().detach()
        img = make_grid(img)
        generate_images.append(img)
        sleep(0.5)
        g_losses.append(g_error.item()/i)   
        d_losses.append(d_error.item()/i)
        tepoch.set_postfix({'G loss (final)': g_error.item()/i, 
                            'D loss (final)': d_error.item()/i})
        sleep(0.1)
        #print('Epoch {}: g_loss: {:.8f} d_loss: {:.8f}\r'.format(epoch, g_error/i, d_error/i))
        
    print('Training Finished')
    torch.save(self.generator.state_dict(), 'mnist_generator.pth')
    return g_losses, d_losses, generate_images
```

Mathematics for training GAN: [here](https://jaketae.github.io/study/gan-math/) or [this video](https://www.youtube.com/watch?v=Gib_kiXgnvA)  

### load MNIST data
### usage

# Computer vision
## limit of multi perceptron

## Preprocess
### point processing
### local processing
### activation function
### downsampling
## vanila CNN
### define CNN object
### prediction
# DCGAN
## upsampling
## ConvTranspose2D
## define DCGAN object
