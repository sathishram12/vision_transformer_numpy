# vision_transformer_numpy
NumPy Implementation of the [Vision Transformer](https://openreview.net/pdf?id=YicbFdNTTy) (ViT) on Num/Cu-py

In order to gain a deeper understanding of Vision Transformers (ViT) and also I didnt see any previous work has demonstrated backward propagation in conjunction with forward propagation,
Therefore, I come up with implementing vision transformer in numpy (cpu)/ cupy(gpu)

Here are the main benefits of implementing ViT in NumPy:
* It aids in comprehending the underlying mathematics, preventing the abstraction of the learning process.
* It eliminates the need for the pytorch framework.

## Dataset:
For sake of simplicity the code uses MNIST dataset as from [here](http://ldaplusplus.com/files/mnist.tar.gz).

## Training
The model trained in the code is currently not saved. Loss and metrics are provided.

### Need to add/implement
- [] Resolve bugs of  overflow errors and occurance of nan values
- [] save model weights 
- [] unit tests
  
