import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT

import wandb
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'validation_accuracy'
        },
    'parameters': {
        "embed_dim": {'value':512},
        "num_heads": {'value':16},
        "num_layers": {'value':2},
        "pool": {'value':"mean"},
        "pos_enc": {'value':"fixed"},
        # "embed_dim": {'values':[128, 256, 512]},
        # "num_heads": {'values':[4, 8, 16, 32]},
        # "num_layers": {'values':[2, 4, 6, 8]},
        # "pool": {'values':["max", "mean", "cls"]},
        # "pos_enc": {'values':["fixed", "learnable"]},
        "num_classes": {'value':2},
        "channels": {'value':3},
        "num_epochs": {'value':10},
        "dropout": {'value':0.3},
        "fc_dim": {'value':None},
        "batch_size": {'value':16},
        "lr": {'value':1e-4},
        "warmup_steps": {'value':625},
        "weight_decay": {'value':1e-3},
        "gradient_clipping": {'value':1},
        }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="ADLinCV_Week2-vision-transformers")


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # select two classes 
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    trainset, _ = torch.utils.data.random_split(trainset, [5000, 5000])
    testset, _ = torch.utils.data.random_split(testset, [1000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    return trainloader, testloader, trainset, testset


def main(config=None):
    set_seed(seed=1)

    wandb.init(config=sweep_configuration, project="ADLinCV_Week2-vision-transformers")
    config = wandb.config

    image_size=(32,32)
    patch_size=(4,4)
    channels=config.channels
    embed_dim=config.embed_dim
    num_heads=config.num_heads
    num_layers=config.num_layers
    num_classes=config.num_classes
    pos_enc=config.pos_enc
    pool=config.pool
    dropout=config.dropout
    fc_dim=config.fc_dim
    num_epochs=config.num_epochs
    batch_size=config.batch_size
    lr=config.lr
    warmup_steps=config.warmup_steps
    weight_decay=config.weight_decay
    gradient_clipping=config.gradient_clipping

    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, 
                patch_size=patch_size, 
                channels=channels, 
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                num_layers=num_layers,
                pos_enc=pos_enc, 
                pool=pool, 
                dropout=dropout, 
                fc_dim=fc_dim, 
                num_classes=num_classes
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # training loop
    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train()
        for image, label in tqdm.tqdm(train_iter):
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for image, label in test_iter:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out = model(image).argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            print(f'-- {"validation"} accuracy {acc:.3}')
        wandb.log({"validation_accuracy": acc})

    # ## Save attention maps for a few images
    # model.eval()
    # for image, label in test_iter:
    #     if torch.cuda.is_available():
    #         image, label = image.to('cuda'), label.to('cuda')
    #     out = model(image)
    #     break
    
    # # Get attention maps when the model has no attention pooling

    
    # fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # fig.suptitle('Attention Maps')
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(attention_maps[0][i].cpu().detach().numpy())
    #     ax.axis('off')
    #     ax.set_title(f'head {i//4+1}')
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # # Save these attention maps for a few images
    # plt.savefig("attention_maps.png")
    # # wandb.log({"attention_maps": [wandb.Image(attention_maps[0][i].cpu().detach().numpy()) for i in range(8)]})

    

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    # main()
    wandb.agent(sweep_id, main, count=1)
