# Introduction

Input data is fed into the neural network - from the left - and it gets forward propagated through it
to produce the output on the other end - right!...

*propagated: added, multiplied, divided, blah blahed.. we'll go over it!*

Mathematically a neural network is simply a pure function with many many parameters.

Let's discuss functions for a bit!

It's common knowledge that a function is a relationship
between a set of possible inputs and a set of possible outputs.

Theory allows the pattern of this mapping to be totally random and useless,
but all practical and useful functions embody a pattern.

For instance, when we look at the plot of $sin(\theta)$ we clearly see a wavy sinusoidal
pattern, which has been useful to us humans for so many years.

Similarly the infamous equation of a line $y = mx + b$ is a function that can predict
the value of some variable $y$ given a variable $x$ as long as:
1. parameters $m$ and $b$ are accurately known
2. $x$ and $y$ have a linear relationship

And that's essentially what linear regression is.

$$\large Everything\ has\ a\ pattern.$$
$$\large Functions\ capture\ patterns.$$
$$\large Complex\ patterns\ are\ captured\ by\ complex\ functions.$$

No surprise now when we think about conversational AIs like ChatGPT.
Its just a very complex - 175 Billion Parameter - function that has pretty
accurately captured the pattern of human language, and not just human language
it has gone one step meta, it has captured patterns that are based on top of
human language: human knowledge.

It's a heartless, mindless, function that receives a body of text - your question, and the conversation before it - $x$, and predicts the next word $y$ given all that it has learnt $m\ and\ b$.
But what it produces is so convincingly human like that we're all freaking out!

*Also if you've noticed when you're interacting with ChatGPT it feels
like its typing one word after another. That's not a UX thing, it's
actually the model predicting one word after another - for nerds: WebSocket. It's given the existing conversation and it produces the next word, and then it's
given the conversation with the new word added and it generates
the next word, and so on until it predicts an "end of text" token,
which is weirdly similar to how we decide when to shut up!*

# What I want to achieve with this corpus of text

It's easy to understand how neural networks produce useful output
once they have been assigned correct parameters -
analogues for $m\ and\ b$. But where the magic lies is in understanding
how we arrive at the correct parameters.

It's a really cool mathematical technique called $back\ propagation$
that allows us to train a neural network and make it go from a
useless function that produces garbage as output to a practically
useful one that captures underlying patterns in the training data and predicts output with accuracy.

The landscape of AI is ruled by gods of Calculus and Linear Algebra.
This text is especially for people who love understanding things from
a first principles perspective, and are willing to love math because that's
the only way this thing will ever make sense. Remember thinking "when will I ever use this?" in Cal and LA classes?? You can use this now.

Here goes another attempt at slaying the BackProp dragon!

# Formalizing Forward Propagation

Before we can start understanding BackProp we must express
forward propagation, mathematically - the process of converting a neural network's
input and producing an output.

From here onwards the discussion will get highly tecnhical,
the readers are expected to have a general idea about weights, biases, artificial neurons.

<!-- do a section for getting beginners upto speed and link it here -->



# Single Neuron

For $k$ features. The weighted input of a single neuron is given by the following equation

$$
z = (\sum_{i=1}^k w_{i} .x_{i})+b \\
$$

The output of this neuron is given by

$$
a = \sigma(z)
$$

where $\sigma$ is the sigmoid function. It can be some other activation function too.


# Structure of the neural network

- Imagined from left to right, with a number of layers.
- Each layer is a vertical column of neurons.
- Leftmost layer is input layer. Each neuron in this layer connects to all the neurons in the next layer through weights.
- Rightmost layer is the output layer. It is fully connected with the layer on its left. Its activations are the output of the network.
- All the layers in the middle are called hidden layers, these layers are fully connected on both sides and they receive the input from the left, they produce activations which get forwarded to the layer on the right.


# Layout of the Weight matrix $w^L$

Each layer stores what is called a weight matrix.
This matrix stores all the weights that connect this layer to
the previous layer - layer on the left.
A weight is a floating point value between 0 and 1,
that decides the strength of a connection between two neurons.
We formally define a weight as:


## $$w_{jk}^L$$ 
Represents a weight that connects $j_{th}$ neuron in layer $L$ - current layer - to $k_{th}$ neuron in layer $(L-1)$ - previous layer.

*Example: $w_{27}^4$ represents a weight that connects $2_{nd}$ neuron in layer $4$ to $7_{th}$ neuron in layer $3$*

<!-- insert picture of a weight between layer L and L-1 -->

Weight matrix for layer $L$ is denoted by
## $$w^L$$
The number of rows in $w^L$ is equal to the number of neurons in layer $L$.
And the number of columns in $w^L$ is equal to the number of neurons in layer
$L-1$

In summary, following is the layout of a weight matrix for layer $L$

$$
w^L =
\begin{bmatrix}
w_{00}^L \cdots w_{0k}^L \\
\vdots \ddots \vdots \\
w_{j0}^L \cdots w_{jk}^L \\
\end{bmatrix}
$$


# Definitions


## $b_{j}^L$ 
### Represents a bias for the $j_{th}$ neuron in layer $L$
*Example: $b_{4}^5$ represents the bias for the $4_{th}$ neuron
in layer $5$*

## $z_{j}^L$ 
### Represents weighted input for neuron $j$ in layer $L$
*Example: $z_{4}^5$ represents the weighted input for the $4_{th}$ neuron
in layer $5$*


## $a_{j}^L$ 
### Represents activation for the $j_{th}$ neuron in layer $L$
*Example: $a_{4}^5$ represents the activation for the $4_{th}$ neuron
in layer $5$*




# Calculations
The weighted input for a neuron $j$ in layer $L$ is calculated like this

## $z_{j}^L = (\sum_{k=0}^k w_{jk}^L .a_{k}^{L-1})+b_{j}^L $

The activation for a neuron $j$ in layer $L$ is calculated like this

## $a_{j}^L = \sigma(z_{j}^L)$


# Vectorized Form

When we're implementing neural networks we rely on vector/matrix
operations to take advantage of SIMD operations available on modern
CPUs and GPUs. Therefore we should express equations for forward/backward
propagation in vectorized forms.

## Weight Matrix

We've already defined the vectorized form of the weights as the weight matrix.
For a layer $L$ the weight matrix is given as:


$$
w^L =
\begin{bmatrix}
w_{00}^L \cdots w_{0k}^L \\
\vdots \ddots \vdots \\
w_{j0}^L \cdots w_{jk}^L \\
\end{bmatrix}
$$

## Bias Vector

For a layer $L$ the bias vector is expressed as:


$$
b^L =
\begin{bmatrix}
b_{0}^L \\
\vdots \\
b_{j}^L \\
\end{bmatrix}
$$

where $j+1$ is the number of neurons in layer $L$

## Weighted Input Vector

For a layer $L$ the weighted input of neurons in layer $L$ is expressed as:

$$ z^L = w^L . a^{L-1} + b^L $$

where:
- $w^L$ is the weight matrix of layer $L$
- $a^{L-1}$ is the activation vector of layer $L-1$
- $b^L$ is the bias vector of layer $L$

## Activation Vector

For a layer $L$ the activation vector is expressed as:

$$ a^L = \sigma(z^L) $$

where:
- $\sigma$ is the activation function - sigmoid in this case.
- $z^l$ is the weighted input vector for layer $L$


# Summary

Given

$w^L$ the weight matrix for layer $L$

$a^{L-1}$ the activation vector for layer $L-1$

$b^L$ the bias vector for layer $L$

$\sigma$ an activation function

we forward propagate by first computing $z^L$

$$ z^L = w^L . a^{L-1} + b^L $$

and then we calculate $a^L$

$$ a^L = \sigma(z^L) $$

this allows us to compute the activations of the next layer $a^{L+1}$
