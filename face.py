
import os
import torch
from tqdm import tqdm
import logging

import numpy as np
import torchvision


class FaceRecognitionModel:

    def __init__(self, backbone, header, loss):
        self.backbone = backbone
        self.header = header
        self.loss = loss

        # join parameter sets of backbone and header
        self.params = list(backbone.parameters())
        self.params.extend(list(header.parameters()))

        #logging.info("")
        print(f"Built FR Model: [{backbone.__class__.__name__} -> {header.__class__.__name__} -> {loss.__class__.__name__}]")


    def fit(self, train_dataset, epochs, batch_size, initial_lr, device, evaluator=None):

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(params=self.params, lr=initial_lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10,18,22], gamma=0.1)

        self.header.to(device)
        self.backbone.to(device)

        for e in range(1, epochs+1):
            train_losses = np.empty(len(train_dataloader))

            self.header.train()
            self.backbone.train()

            pbar = tqdm(enumerate(train_dataloader))
            for step, batch in pbar:

                inputs = batch['image'].to(device)
                labels = batch['label'].to(device).long().view(-1)

                features = self.backbone(inputs)
                features = features.view(features.shape[0], features.shape[1])

                outputs = self.header(features, labels)

                loss_value = self.loss(outputs, labels)
                train_losses[step] = loss_value

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                pbar.set_description_str(f"[{e}/{epochs}]({step+1}/{len(train_dataloader)}) - Train_Loss: {train_losses[:step+1].mean()}")

            if evaluator is not None:
                evaluator(model=self.backbone, epoch=e, device=device)

            if lr_scheduler is not None:
                lr_scheduler.step()


    def encode(self, imgs, device):
        batch = [torchvision.transforms.functional.to_tensor(img) for img in imgs]
        batch = torch.stack(batch).to(device)
        return self.backbone(batch).cpu().detach().numpy()
