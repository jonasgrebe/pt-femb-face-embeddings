# femb - Small Face Embedding Library

```python
backbone = build_backbone(backbone="iresnet18", embed_dim=embed_dim)
header = ArcFaceHeader(in_features=embed_dim, out_features=train_n_classes)
loss = torch.nn.CrossEntropyLoss()

face_model = FaceEmbeddingModel(backbone=backbone, header=header, loss=loss)
```

#### Basic Framework:
+ **Backbone**: The actual embedding network that we want to train. It takes some kind of input and produces a feature representation (embedding) of a certain dimensionality.
+ **Header**: A training-only extension to the backbone network that is used to predict the identity class logits for the loss function. This is the main part where the implemented methods (SphereFace, CosFace, ...) differ.
+ **Loss**: The loss function that is used to judge how good the (manipulated) logits are. Usually, this is the cross-entropy loss.

```python
evaluator = VerificationEvaluator(similarity='cos')
optimizer = torch.optim.SGD(params=face_model.params, lr=1e-1, momentum=0.9, weight_decay=5e-4)

lr_global_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[8000, 10000, 160000], gamma=0.1)

# fit the face embedding model to the dataset
face_model.fit(
    train_dataset=train_dataset,       # specify the training set
    batch_size=32,                     # batch size for training and evaluation
    device='cuda',                     # torch device, i.e. 'cpu' or 'cuda'
    optimizer=optimizer,               # torch optimizer
    lr_epoch_scheduler=None,           # scheduler based on epochs
    lr_global_step_scheduler=None,     # scheduler based on global steps
    evaluator=evaluator,               # evaluator module
    val_dataset=val_dataset,           # specify the validation set
    evaluation_steps=10,               # number of steps between evaluations
    max_training_steps=20000,          # maximum number of (global) training steps (if zero then max_epochs count is used for stopping)
    max_epochs=0,                      # maximum number of epochs (if zero then max_training_steps is used for stopping)
    tensorboard=True                   # specify whether or not tensorboard shall be used for embedding projections
    )
```


#### Implemented Losses
+ SphereFace Loss
+ CosFace Loss
+ ArcFace Loss
+ MagFace Loss
