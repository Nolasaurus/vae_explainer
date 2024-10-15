# Variational AutoEncoders

## Section 1: Regular AutoEncoders
### Why do I care about AutoEncoders?
- useful for noise reduction in audio, images, and video (e.g. https://arxiv.org/abs/2303.00656)
- useful for anomaly detection
- data compression
- dimensionality reduction
- feature learning
### What is an AutoEncoder?
- auto: "self", encod-:
- latent space representation, z
- Encoder, Decoder, Architecture
- Pic showing the encoder neural net mapping the input image (x) to a smaller latent space representation z, then being decoded by neural net (an inverse of the encoder) to reconstruct the input image (x_prime)(i.e. mapped back to image space).
- ![Basic architecture of an autoencoder](https://github.com/madebyollin/taesd/blob/main/images/reconstruction_example.jpg)
- x_prime := Dec(Enc(x)
- x_prime is the result of mapping the input image to a point in the latent space, then reconstructing the image
#### Example 1: train autoencoder to recreate transformation of image (RGB-> Grayscale, Rescale(50%))
- img_transform = v2.Compose(v2.Grayscale(img), v2.Rotate(90), v2.Rescale(0.5))
- x_orig = img(path='folder') # input images
- x_transform = img_transform(x_orig) # image transformation
- x_prime = dec(enc(x_orig)
- calc_MSE_loss(x_prime, x_transform) # use x_transform to calculate loss
- compare compute and memory requirements
  - img_transform(x_orig) -> x_transform
  - x_orig -> z -> x_prime
- the neural nets are able to capture a complicated pipeline of affine and color space transformations.

### How Does Training Work?
- Training Objective: The AutoEncoder is trained to minimize the difference between the input (x) and the output reconstruction (x').
- Loss function: Mean Squared Error (MSE) (sometimes use Binary Cross-Entropy (BCE) instead)
- Optimization Process: Backpropagation and gradient descent are used to update the weights of both the encoder and decoder networks during training.
- Challenges: Discuss challenges like overfitting, where the model learns to memorize the input instead of generalizing to unseen data.

#### Example 2: Train AE with labeled classifier dataset, CIFAR10 from Caltech. (CIFAR10(root[, train, transform, ...])
- Train autoencoder with dataset
- What happens if we pass an image that's totally unlike what we used to train?
- Overfitting to training dataset (poorly generalizes to unseen data)
- What if we wanted to get a "general" reconstruction of one of the categories? It's "seen" 10s of thousands of examples, can't just grab a point in the middle of all of the latent representations and get a new example of the category?
- Not with regular autoencoders, the latent space is not continuous, meaning you can't grab a point near one of a training image and get a good reconstruction of something like the training image.
##### Show example of poor reconstruction
- Calculate multi-dimensional object created by train dataset, grab a bunch of points near the center
- Use nearest neighbors to interpolate points between groups
##### It doesn't work very well
- Segue to variational auto-encoders


## Section 2: Variational AutoEncoders (for next version)




