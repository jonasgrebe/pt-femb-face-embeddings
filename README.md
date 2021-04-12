# femb - Small Face Embedding Library

#### Basic Framework:
+ **Backbone**: The actual embedding network that we want to train. It takes some kind of input and produces a feature representation (embedding) of a certain dimensionality.
+ **Header**: A training-only extension to the backbone network that is used to manipulate the predicted values for the loss function. This is the main part where the implemented methods (SphereFace, CosFace, ...) differ.
+ **Loss**: The loss function that is used to judge how good the (manipulated) embeddings are. Usually, this is the cross-entropy loss.
 
#### Implemented Losses
+ SphereFace Loss
+ CosFace Loss
+ ArcFace Loss
+ MagFace Loss

#### Implemented Dataset Classes:
+ MNIST
+ CelebA
+ LFW
+ (CASIA Webface)

#### Implemented Evaluation Procedures
+ LFW Verification
