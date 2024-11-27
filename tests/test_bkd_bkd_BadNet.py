import os
import sys
import time
sys.path.append("/localtmp/xxxx/BackdoorBox")

# sys.path.append("/home/xxxx/BackdoorBox")
# sys.path.append("/home/xxxx/BackdoorVault")
# sys.path.append("/home/xxxx/Circumventing-Backdoor-Defenses")
from utils import DatasetNumpy
from scipy.fftpack import dct, idct

import random

import PIL
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from CelebAOwn import CelebAOwn
from core.defenses.STRIP import STRIP
from core.defenses.Frequency import Frequency
# from core.defenses.Lava_D import LAVA

import argparse
import core

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import Subset

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
import cv2
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE

import time
class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label


def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError

def gen_grid(height, k, intensity = 1):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = intensity * noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

def prepare_dataset(args):


    if args.dataset == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
        img_width = 32
        img_height = 32

        transform_train = Compose([
            Resize((img_width, img_height)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = Compose([
            Resize((img_width, img_height)),
            ToTensor()
        ])
        clean_trainset = dataset(args.datasets_root_dir, train=True, transform=transform_train, download=True)
        clean_testset = dataset(args.datasets_root_dir, train=False, transform=transform_test, download=True)


        target_label = 0
        poisoned_transform_train_index = 0
        poisoned_transform_test_index = 0

    elif args.dataset == "TinyImagenet":
        datasets_root_dir = "tiny-imagenet-200"

        if "EfficientNet" in args.model_name:


            transform_train = Compose([



                transforms.ToTensor(),
                # transforms.Resize(224),  # Resize images to 256 x 256
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_train = Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(), # for cifar10
            ])

        import cv2

        clean_trainset = DatasetFolder(root=os.path.join(datasets_root_dir, 'train'),
                                 transform=transform_train,
                                 loader=cv2.imread,
                                 extensions=('jpeg',),
                                 target_transform=None,
                                 is_valid_file=None,
                                 )
        if "EfficientNet" in args.model_name:
            print("================")
            transform_test = Compose([
                ToTensor(),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
                # transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_test = Compose([
                ToTensor(),
            ])

        clean_testset = DatasetFolder(root=os.path.join(datasets_root_dir, 'val'),
                                transform=transform_test,
                                loader=cv2.imread,
                                extensions=('jpeg',),
                                target_transform=None,
                                is_valid_file=None,
                                )
        img_width = 64
        img_height = 64

        target_label = 0
        poisoned_transform_train_index = 0
        poisoned_transform_test_index = 0

    elif args.dataset == "CelebAOwn":
        data_root = "datasets"

        img_width, img_height = 64, 64

        transform_train = Compose([
            Resize((img_width, img_height)),
            ToTensor()
        ])

        clean_trainset = CelebAOwn(data_root, target_type="attr", split='train', transform=transform_train,
                           download=False, size=-1)

        transform_test = Compose([
            Resize((img_width, img_height)),
            ToTensor()
        ])

        clean_testset = CelebAOwn(data_root, split='test', transform=transform_test, download=False, size=-1)
        # test_dataset = torchvision.datasets.CelebA(data_root, split="test", target_type=["attr", "landmarks"], transform=transforms)

        target_label = 0
        poisoned_transform_train_index = 0
        poisoned_transform_test_index = 0

    elif args.dataset == "GTSRB":
        import os.path as osp
        import cv2

        img_width, img_height = 32, 32

        datasets_root_dir = "./data"


        if args.attack_method in ["BadNet", "Blend", "ISSBA"]:
            #BadNet
            transform_train = Compose([
                ToPILImage(),
                Resize((img_width, img_height)),
                ToTensor()
            ])

            transform_test = Compose([
                ToPILImage(),
                Resize((img_width, img_height)),
                ToTensor()
            ])

            target_label = 1
            poisoned_transform_train_index = 2
            poisoned_transform_test_index = 2

        elif args.attack_method in ["WaNet"]:

            # WaNet
            transform_train = Compose([
                ToTensor(),
                RandomHorizontalFlip(),
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                ToTensor()
            ])


            transform_test = Compose([
                ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                ToTensor()

            ])
            poisoned_transform_train_index = 0
            poisoned_transform_test_index = 0
        else:
            raise NotImplementedError

        clean_trainset = DatasetFolder(
            root=osp.join(datasets_root_dir, 'GTSRB', 'Train'),  # please replace this with path to your training set
            loader=cv2.imread,
            extensions=('png',),
            transform=transform_train,
            target_transform=None,
            is_valid_file=None)




        clean_testset = DatasetFolder(
            root=osp.join(datasets_root_dir, 'GTSRB', 'testset'),  # please replace this with path to your test set
            loader=cv2.imread,
            extensions=('png',),
            transform=transform_test,
            target_transform=None,
            is_valid_file=None)

    elif args.dataset == "ImageNet_Subset":
        import cv2
        datasets_root_dir = "../../datasets/imagenette2-160"
        img_width, img_height = 224, 224
        transform_train = Compose([
            ToPILImage(),
            RandomHorizontalFlip(),
            Resize((img_width, img_height)),
            ToTensor()
        ])
        transform_test = Compose([
            ToPILImage(),
            Resize((img_width, img_height)),
            ToTensor()
        ])

        clean_testset = DatasetFolder(root=os.path.join(datasets_root_dir, 'val'),
                                  transform=transform_test,
                                  loader=cv2.imread,
                                  extensions=('jpeg',),
                                  target_transform=None,
                                  is_valid_file=None,
                                  )
        clean_trainset = DatasetFolder(root=os.path.join(datasets_root_dir, 'train'),
                                       transform=transform_train,
                                       loader=cv2.imread,
                                       extensions=('jpeg',),
                                       target_transform=None,
                                       is_valid_file=None,
                                       )

        target_label = 0
        poisoned_transform_train_index = 3
        poisoned_transform_test_index = 2


    else:
        raise NotImplementedError("Please specify a dataset name")

    if args.attack_method == "BadNet":

        pattern = torch.zeros((img_width, img_height), dtype=torch.uint8)

        # for CelebA

        # pattern[-int(img_width * 0.1):, -int(img_width * 0.1):] = torch.rand((int(img_width * 0.1), int(img_width * 0.1))) * 255

        # pattern[-int(img_width * 0.1):, -int(img_width * 0.1):] = 0


        # for Cifar10
        if args.dataset in ["Cifar10", "GTSRB"]:
            pattern[-3, -3] = 255
            pattern[-3, -2] = 0
            pattern[-3, -1] = 255
            pattern[-2, -3] = 0
            pattern[-2, -2] = 255
            pattern[-2, -2] = 0
            pattern[-1, -3] = 255
            pattern[-1, -2] = 0
            pattern[-1, -1] = 255
        elif args.dataset in ["ImageNet_Subset_"]:
            pattern[-int(img_width * 0.1):, -int(img_width * 0.1):] = 255
        else:

            pattern[-int(img_width * args.trigger_size):, -int(img_width * args.trigger_size):] = torch.rand(
                (int(img_width * args.trigger_size), int(img_width * args.trigger_size))) * 255

        # pattern[-int(img_width * args.trigger_size):, -int(img_width * args.trigger_size):] = 255

        # pattern[-int(img_width * args.trigger_size):, -int(img_width * args.trigger_size):] = torch.rand(
        #     (int(img_width * args.trigger_size), int(img_width * args.trigger_size))) * 255

        weight = torch.zeros((img_width, img_height), dtype=torch.float32)
        weight[-int(img_width * args.trigger_size):, -int(img_width * args.trigger_size):] = 1.0

        attack = core.BadNets(
            train_dataset=clean_trainset,
            test_dataset=clean_testset,
            model=core.models.ResNet(18),
            loss=nn.CrossEntropyLoss(),
            y_target=target_label,
            poisoned_rate=args.poisoned_rate,
            pattern=pattern,
            weight=weight,
            poisoned_transform_train_index = poisoned_transform_train_index, # GTSRB
            poisoned_transform_test_index = poisoned_transform_test_index, # GTSRB
            clean_label=args.clean_label
        )
        poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()




        # for idx in range(20):
        #     plt.imshow(poisoned_trainset[idx][0].permute(1,2,0))
        #     plt.show()
        #     input()
    elif args.attack_method == "LC":
        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': "1",
            'GPU_num': 1,

            'benign_training': False,  # Train Attacked Model
            'batch_size': 128,
            'num_workers': 8,

            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'schedule': [150, 180],

            'epochs': 200,

            'log_iteration_interval': 100,
            'test_epoch_interval': 10,
            'save_epoch_interval': 10,

            'save_dir': 'experiments',
            'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
        }

        pattern = torch.zeros((32, 32), dtype=torch.uint8)
        pattern[-1, -1] = 255
        pattern[-1, -3] = 255
        pattern[-3, -1] = 255
        pattern[-2, -2] = 255

        pattern[0, -1] = 255
        pattern[1, -2] = 255
        pattern[2, -3] = 255
        pattern[2, -1] = 255

        pattern[0, 0] = 255
        pattern[1, 1] = 255
        pattern[2, 2] = 255
        pattern[2, 0] = 255

        pattern[-1, 0] = 255
        pattern[-1, 2] = 255
        pattern[-2, 1] = 255
        pattern[-3, 0] = 255

        weight = torch.zeros((32, 32), dtype=torch.float32)
        weight[:3, :3] = 1.0
        weight[:3, -3:] = 1.0
        weight[-3:, :3] = 1.0
        weight[-3:, -3:] = 1.0

        eps = 8
        alpha = 1.5
        steps = 100
        max_pixel = 255

        print(transform_train)
        attack = core.LabelConsistent(
            train_dataset=clean_trainset,
            test_dataset=clean_testset,
            model=core.models.ResNet(18),
            adv_model=core.models.ResNet(18),
            adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{args.poisoned_rate}_seed{global_seed}',
            loss=nn.CrossEntropyLoss(),
            y_target=target_label,
            poisoned_rate=0.3,
            pattern=pattern,
            weight=weight,
            eps=eps,
            alpha=alpha,
            steps=steps,
            max_pixel=max_pixel,
            poisoned_transform_train_index=0,
            poisoned_transform_test_index=0,
            poisoned_target_transform_index=0,
                schedule=schedule,
            seed=global_seed,
            deterministic=True
        )
        # attack.train()
        # input()
        poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
    elif args.attack_method == "Refool":
        # load reflection images
        reflection_images = []
        reflection_data_dir = "/home/xxxx/BackdoorBox/data/VOCdevkit/VOC2012/JPEGImages/"  # please replace this with path to your desired reflection set
        reflection_image_path = os.listdir(reflection_data_dir)
        reflection_images = [read_image(os.path.join(reflection_data_dir, img_path)) for img_path in
                             reflection_image_path[:200]]
        attack = core.Refool(
            train_dataset=clean_trainset,
            test_dataset=clean_testset,
            model=core.models.ResNet(18),
            loss=nn.CrossEntropyLoss(),
            y_target=target_label,
            poisoned_rate=args.poisoned_rate,
            poisoned_transform_train_index=poisoned_transform_train_index,
            poisoned_transform_test_index=poisoned_transform_test_index,
            poisoned_target_transform_index=0,
            schedule=None,
            seed=global_seed,
            deterministic=True,
            reflection_candidates=reflection_images,
        )
        poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

    elif args.attack_method == "WaNet":
        identity_grid, noise_grid = gen_grid(img_width, int(img_height/8), 50 if args.dataset == "ImageNet_Subset" else 1)
        attack = core.WaNet(
            train_dataset=clean_trainset,
            test_dataset=clean_testset,
            model=None,
            loss=None,
            y_target=0,
            poisoned_rate=args.poisoned_rate,
            identity_grid=identity_grid,
            noise_grid=noise_grid,
            noise=False,
            poisoned_transform_train_index=poisoned_transform_train_index, 
            poisoned_transform_test_index=poisoned_transform_test_index
        )
        poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()



    elif args.attack_method == "Blend":
        # pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
        # pattern[0, -3:, -3:] = 255
        import cv2
        pattern = cv2.imread('image.png')
        pattern = torch.from_numpy(cv2.resize(pattern, (img_width, img_height))).permute(2, 0, 1)
        print(pattern.shape)

        weight = torch.zeros((1, img_width, img_height), dtype=torch.float32)
        # weight[:, int(0.2 * img_width):, int(0.2 * img_height):] = 0.8
        weight[:, :int(img_width), :int(img_height)] = 0.2
        attack = core.Blended(
            train_dataset=clean_trainset,
            test_dataset=clean_testset,
            model=core.models.ResNet(18),
            loss=nn.CrossEntropyLoss(),
            pattern=pattern,
            weight=weight,
            y_target=1,
            poisoned_rate=args.poisoned_rate,
            seed=global_seed,
            deterministic=True,
            poisoned_transform_train_index=poisoned_transform_train_index,  # GTSRB
            poisoned_transform_test_index=poisoned_transform_test_index  # GTSRB
        )
        poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

    elif args.attack_method == "ISSBA":

        if not os.path.exists("ISSBA_poisoned_trainset_{}.pth".format(args.dataset)):
            secret_size = 20

            train_data_set = []
            train_secret_set = []
            for idx, (img, lab) in enumerate(clean_trainset):
                train_data_set.append(img.tolist())
                secret = np.random.binomial(1, .5, secret_size).tolist()
                train_secret_set.append(secret)

            for idx, (img, lab) in enumerate(clean_testset):
                train_data_set.append(img.tolist())
                secret = np.random.binomial(1, .5, secret_size).tolist()
                train_secret_set.append(secret)

            train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)

            encoder_schedule = {
                'secret_size': secret_size,
                'enc_height': 32,
                'enc_width': 32,
                'enc_in_channel': 3,
                'enc_total_epoch': 20,
                'enc_secret_only_epoch': 2,
                'enc_use_dis': False,
            }

            schedule = {
                'device': 'GPU',
                'GPU_num': 1,

                'benign_training': False,
                'batch_size': 128,
                'num_workers': 8,

                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [150, 180],

                'epochs': 0,

                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
                'save_epoch_interval': 100,

                'save_dir': 'experiments',
                'experiment_name': 'train_poison_DataFolder_CIFAR10_ISSBA',

                # 'pretrain': "ResNet18_ISSBA.pth"
            }



            attack = core.ISSBA(
                dataset_name="Cifar10", # to avoid normalizer
                train_dataset=clean_trainset,
                test_dataset=clean_testset,
                train_steg_set=train_steg_set,
                model=core.models.ResNet(18),
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=args.poisoned_rate,  # follow the default configure in the original paper
                encoder_schedule=encoder_schedule,
                encoder=None,
                seed=global_seed,
                schedule=schedule
            )


            attack.train(schedule=schedule)

            poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

            torch.save(poisoned_trainset, "ISSBA_poisoned_trainset_{}.pth".format(args.dataset))
            torch.save(poisoned_testset, "ISSBA_poisoned_testset_{}.pth".format(args.dataset))
        else:
            poisoned_trainset = torch.load("ISSBA_poisoned_trainset_{}.pth".format(args.dataset))
            poisoned_testset = torch.load("ISSBA_poisoned_testset_{}.pth".format(args.dataset))

            # for img, target in poisoned_trainset:
            #     print(target, type(poisoned_trainset))
            #     break

    else:
        raise NotImplementedError("Please specify an attack method")

    return clean_trainset, clean_testset, poisoned_trainset, poisoned_testset

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')
def three_channel_dct( x):
    dct_data=torch.zeros(x.shape).cuda()
    for i in range(x.shape[0]):
        for channel in range(3):
            dct_data[i][channel,:, : ] = torch.tensor(dct2((x[i][channel,:, : ] * 255).detach().cpu().numpy().astype(np.uint8)))
    return torch.tensor(dct_data)

def train_detector(args,train_dataset, poisoned_testset, clean_testset, structure=50, class_num=100, input_size=32,label_number=10, num_epochs=5, batch_size=128, learning_rate=0.05, weight_decay = 1e-4  # Note: This is used in Keras as L2 regularization, handled differently in PyTorch
):
    # Model initialization using gpu if available

    model = core.models.backdoor_backdoor.detector_model(structure, class_num, input_size,label_number)
    model.train()
    model = model.cuda()

    #model parallel
    # model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    # Check if model checkpoint exists
    # model_path = "{}_{}_{}_{}.pth".format(args.model_name,args.attack_method,learning_rate,num_epochs)
    model_path = "{}_{}_{}_{}.pth".format(args.model_name,args.attack_method,learning_rate,num_epochs)

    # model_path = "{}_{}_{}_{}_{}.pth".format(args.model_name,args.attack_method,args.dataset,learning_rate,num_epochs)
    print(model_path)


    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        return model


    running_loss = 0.0
    model.train()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Data preparation
    last_time = time.time()

    for epoch in tqdm(range(num_epochs)):        # Convert to PyTorch tensors
        # Create DataLoader
    
        # Training loop
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data=data.cuda()
            data_dct= three_channel_dct(data).cuda()

            output = model(data, data_dct)

            loss = criterion(output, target.cuda())
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 50 == 0: 
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                print(msg)
                 # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(data):.3f}')
                running_loss = 0.0
        evaluate(model, poisoned_testset, mode="poisoned")
        evaluate(model, clean_testset, mode="clean")
        model.train()
        scheduler.step()
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("saved")
    return model

def evaluate_bb(model, dataset,image_dct, alpha=0, mode="Poisoned",use_clean_label=False,clean_testset=None,exclude_target=False,target_label=0):
    correct = 0
    total = 0
    testloader = DataLoader(dataset, batch_size=512) #1024 for cifar10
    labels_gt = []
    if use_clean_label or exclude_target:
        testloader_clean=DataLoader(clean_testset, batch_size=512)
        for data, label in testloader_clean:
            labels_gt.append(label)
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs



    if use_clean_label:
        with torch.no_grad():
            for idx, (data) in enumerate(tqdm(testloader)):
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                labels = labels_gt[idx].cuda()
                # labels = clean_data[1].cuda()
                outputs = model(images, image_dct.repeat(images.shape[0], 1, 1, 1)).cuda()
                _, predicted = torch.topk(outputs.data, 2, 1)
                predicted = predicted[:, 0]
                # print(predicted[:20])
                total += labels.size(0)

                correct += (predicted == labels).sum().item()
    else:
        with torch.no_grad():
            for idx, (data) in enumerate(tqdm(testloader)):
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                # print("--",images.shape)
                outputs = model(images,image_dct.repeat(images.shape[0],1,1,1)).cuda()
                _, predicted = torch.topk (outputs.data, 2, 1)

                predicted = predicted[:, 0]

                if exclude_target:
                    # print("===",idx)

                    labels_gt_label = labels_gt[idx].cuda()

                    labels_final=labels[labels_gt_label!=target_label]
                    predicted=predicted[labels_gt_label!=target_label]
                    total += labels_final.size(0)
                    correct += (predicted == labels_final).sum().item()
                    # print("exclude target",total)
                    # print("correct target",correct)
                else:
                    total+=labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # print("exclude target",total)
                    # print("correct target",correct)


    print(f'Accuracy of the network on the ' + mode + f' images: {100 * correct // total} %')
    return 100 * correct // total


def evaluate(model, dataset, alpha=0, mode="Poisoned"):
    correct = 0
    total = 0
    testloader = DataLoader(dataset, batch_size=16) #1024 for cifar10
    # for (img, target) in testloader:
    #     tvu.save_image(img, "temp.png")
    #     input()
    #     break
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            images = alpha * torch.rand(images.shape, device=images.device) + images
            # calculate outputs by running images through the network
            image_dct= three_channel_dct(images).cuda()
            # print(image_dct.shape)
            outputs = model(images,image_dct)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.topk(outputs.data, 2, 1)

            predicted = predicted[:, 0]
            # print(predicted[:20])
            total += labels.size(0)

            correct += (predicted == labels).sum().item()


    print(f'Accuracy of the network on the ' + mode + f' images: {100 * correct // total} %')

def extract_dataset(args):
    clean_testset = DatasetNumpy(args.existing_dataset_path + "clean_testset.npy", args.existing_dataset_path + "_clean_testset")
    poisoned_testset = DatasetNumpy(args.existing_dataset_path + "poisoned_testset.npy", args.existing_dataset_path + "_poisoned_testset")

    return clean_testset, poisoned_testset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='BackdoorBackdoor',
                    description='')
    parser.add_argument("--dataset", default="Cifar10")
    parser.add_argument("--attack_method", default="BadNet")
    parser.add_argument("--datasets_root_dir",  default= '../datasets')
    parser.add_argument("--model_name", default="BackdoorBackdoor")
    parser.add_argument("--epoch_number", type=int, default=200)
    parser.add_argument("--poisoned_rate", type=float, default=0.1)
    parser.add_argument("--trigger_size", type=float, default=0.1)
    parser.add_argument("--existing_dataset", type=bool, default=False)
    parser.add_argument("--use_existing_model", type=bool, default=False)
    parser.add_argument("--visual_latent", type=bool, default=False)
    parser.add_argument("--existing_dataset_path", type=str, default="")
    parser.add_argument("--clean_label", type=bool, default=False)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--structure", type=int, default=18)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1314)
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    print(use_cuda)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device_ids = [0,1,2,3]

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
    gpu = use_cuda


    global_seed = 0

    torch.manual_seed(global_seed)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1314)
    print(use_cuda)

    if not args.existing_dataset:
        clean_trainset, clean_testset, poisoned_trainset, poisoned_testset = prepare_dataset(args)
    else:
        clean_trainset, _, poisoned_trainset, _ = prepare_dataset(args)
        clean_testset, poisoned_testset = extract_dataset(args)

    # def three_channel_dct_one_image(x):
    #     for channel in range(3):
    #         x[channel, :, :] = torch.tensor(
    #             dct2((x[channel, :, :] * 255).detach().cpu().numpy().astype(np.uint8)))
    #     return torch.tensor(x)
    def three_channel_dct_one_image(x):
        dct_data=torch.zeros(x.shape)
        for channel in range(3):
            dct_data[channel, :, :] = torch.tensor(
                dct2((x[channel, :, :] * 255).detach().cpu().numpy().astype(np.uint8)))
        return torch.tensor(dct_data)
    
    import torchvision.utils as tvu
    # tvu.save_image(poisoned_testset[0][0], "./visualization/temp0_badnet_p.png")
    # tvu.save_image(three_channel_dct_one_image(poisoned_testset[0][0]), "./visualization/temp0_frequency_p.png")
    # tvu.save_image(poisoned_testset[1][0], "./visualization/temp1_badnet_p.png")
    # tvu.save_image(three_channel_dct_one_image(poisoned_testset[1][0]), "./visualization/temp1_frequency_p.png")
    # tvu.save_image(poisoned_testset[2][0], "./visualization/temp2_badnet_p.png")
    # tvu.save_image(three_channel_dct_one_image(poisoned_testset[2][0]), "./visualization/temp2_frequency_p.png")
    # tvu.save_image(poisoned_testset[3][0], "./visualization/temp3_badnet_p.png")
    # tvu.save_image(three_channel_dct_one_image(poisoned_testset[3][0]), "./visualization/temp3_frequency_p.png")
    # tvu.save_image(poisoned_testset[4][0], "./visualization/temp4_badnet_p.png")
    # tvu.save_image(three_channel_dct_one_image(poisoned_testset[4][0]), "./visualization/temp4_frequency_p.png")


    # tvu.save_image(clean_testset[0][0], "./visualization/temp0_badnet_c.png")
    # tvu.save_image(three_channel_dct_one_image(clean_testset[0][0]), "./visualization/temp0_frequency_c.png")
    # tvu.save_image(clean_testset[1][0], "./visualization/temp1_badnet_c.png")
    # tvu.save_image(three_channel_dct_one_image(clean_testset[1][0]), "./visualization/temp1_frequency_c.png")
    # tvu.save_image(clean_testset[2][0], "./visualization/temp2_badnet_c.png")
    # tvu.save_image(three_channel_dct_one_image(clean_testset[2][0]), "./visualization/temp2_frequency_c.png")
    # tvu.save_image(clean_testset[3][0], "./visualization/temp3_badnet_c.png")
    # tvu.save_image(three_channel_dct_one_image(clean_testset[3][0]), "./visualization/temp3_frequency_c.png")
    # tvu.save_image(clean_testset[4][0], "./visualization/temp4_badnet_c.png")
    # tvu.save_image(three_channel_dct_one_image(clean_testset[4][0]), "./visualization/temp4_frequency_c.png")
        # input()

    # model = train_model(args, poisoned_trainset, poisoned_testset, clean_testset, use_saved=True)
    model =  train_detector(args,poisoned_trainset, poisoned_testset, clean_testset, structure=args.structure, class_num=100, input_size=args.input_size,
                   label_number=10, num_epochs=100, batch_size=128, learning_rate=0.1, weight_decay=1e-4)


    #The average dct version across the clean dataset
    train_loader = DataLoader(clean_testset, batch_size=100, shuffle=True)
    ps_train_loader = DataLoader(poisoned_testset, batch_size=100, shuffle=True)

    import torchvision.utils as tvu
    for i, (dct_data, target) in enumerate(train_loader):
        
        data_dct = three_channel_dct(dct_data).cuda()
        break
    for i, (dct_data, target) in enumerate(ps_train_loader):
        data_dct_ps = three_channel_dct(dct_data).cuda()
        break

    #visualize the latent space:      
        
    if args.visual_latent:
        print("start visualizing the latent space")
        model.eval()

        embeddings=[]

        clean_labels = []
        poison_labels = []
        args.clean_label = True
        clean_trainset, clean_testset, poisoned_trainset, poisoned_testset = prepare_dataset(args)
        clean_loader_vis = DataLoader(clean_testset, batch_size=1000, shuffle=True)
        poisoned_loader_vis = DataLoader(poisoned_testset, batch_size=1000, shuffle=True) 
        for i, (data, target) in enumerate(clean_loader_vis):
            with torch.no_grad():
                data = data.cuda()
                data_dct = three_channel_dct(data).cuda()
                # print("--",data.shape)
                output = model(data, data_dct)
                # print("here",output.shape)
                normal_emb = model.normal_model(data)
                attack_emb = model.attack_model(data_dct)

            embeddings.append(normal_emb)
            embeddings.append(attack_emb)
            clean_labels.extend(target)
            clean_labels.extend(target)
              
            break

        for i, (data, target) in enumerate(poisoned_loader_vis):
            with torch.no_grad():
                data = data.cuda()
                data_dct = three_channel_dct_one_image(clean_testset[10][0]).repeat(data.shape[0], 1, 1, 1).cuda()
                normal_emb = model.normal_model(data)
                attack_emb = model.attack_model(data_dct)

            embeddings.append(normal_emb)
            embeddings.append(attack_emb)
            poison_labels.extend(target)
            poison_labels.extend(target)
            break
        embeddings = torch.cat(embeddings, dim=0)
        clean_labels = torch.tensor(clean_labels)
        poison_labels = torch.tensor(poison_labels)


        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(embeddings.cpu().numpy())

        import matplotlib.pyplot as plt

        unique_labels = sorted(set(clean_labels.numpy()))
        num_labels = len(unique_labels)
        print("here, the number of labels are", num_labels)
        print("here the number of clean labels are",len(clean_labels))

        cmap = plt.get_cmap('jet', num_labels)
        color=["#8ECFC9","#82B0D2","#8ECFC9","#FA7F6F","#BEB8DC"]
        fig, ax=plt.subplots(figsize=(5,5))
        # fig, ax = plt.subplots()
        ax.spines[['right', 'top']].set_visible(False)
        # ax.axis("off")
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels   # fontsize of the tick labels
        plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

       # ===========================where is the  poisoned data points=========================
        for idx, label in enumerate(unique_labels[:5]):
            subset = reduced_data[:int(0.5*len(clean_labels))][clean_labels.numpy()[:int(0.5*len(clean_labels))] == label]
            plt.scatter(subset[:, 0], subset[:, 1], color=color[idx], marker='o', s=13,alpha=0.6,label="clean")
            subset = reduced_data[int(0.5*len(clean_labels)):len(clean_labels)][clean_labels.numpy()[int(0.5*len(clean_labels)):] == label]
            plt.scatter(subset[:, 0], subset[:, 1], color=color[idx], marker='o', s=13,alpha=0.6,label="clean")
            subset = reduced_data[len(clean_labels):len(clean_labels)+int(0.5*len(poison_labels))][poison_labels.numpy()[:int(0.5*len(poison_labels))] == label]
            if len(subset) > 0:
                plt.scatter(subset[:, 0], subset[:, 1], color=color[idx], marker="d", s=13,alpha=0.8,label="Poisoned")
            subset = reduced_data[len(clean_labels)+int(0.5*len(poison_labels)):len(clean_labels)+len(poison_labels)][poison_labels.numpy()[int(0.5*len(poison_labels)):] == label]
            if len(subset) > 0:
                # print(len(subset))
                plt.scatter(subset[:, 0], subset[:, 1], color=color[idx], marker="d", s=13,alpha=0.8,label="Poisoned")
            # subset = reduced_data[len(clean_labels):][poison_labels.numpy() == label]
            # if len(subset) > 0:
            #     plt.scatter(subset[:, 0], subset[:, 1], cmap=cmap, marker='x', s=13,alpha=0.8, label=f"{clean_testset.classes[label]} poisoned")
        plt.xlim((-60, 80))
        plt.ylim((-80, 80))
        # hfont = {'fontname':'Times New Roman'}
        plt.xlabel("Embedding 1")
        plt.ylabel("Embedding 2")
        plt.title("Embeddings of clean/poisoned images with respect based on PRN and AIN on CIFAR-10", loc='center', wrap=True)
        plt.legend(["clean","poisoned"] ,loc ="upper right",fontsize="8", frameon=False ,borderaxespad=0.2, # Legend closer to the border
            handletextpad=0.1,
            markerscale=1.5,    )
        plt.savefig("cifar_poisoned.png")

        plt.savefig("cifar_poisoned.pdf")
        plt.show()
    

        #==============where is the poisoned data points after using our defense=========================
        for idx, label in enumerate(unique_labels[:5]):
            subset = reduced_data[:int(0.5*len(clean_labels))][clean_labels.numpy()[:int(0.5*len(clean_labels))] == label]
            plt.scatter(subset[:, 0], subset[:, 1], color=color[idx], marker='o', s=13,alpha=0.6,label="clean")
            subset = reduced_data[int(0.5*len(clean_labels)):len(clean_labels)][clean_labels.numpy()[int(0.5*len(clean_labels)):] == label]
            plt.scatter(subset[:, 0], subset[:, 1], color=color[idx], marker='o', s=13,alpha=0.6,label="clean")
            subset = reduced_data[len(clean_labels):len(clean_labels)+int(0.5*len(poison_labels))][poison_labels.numpy()[:int(0.5*len(poison_labels))] == label]
            if len(subset) > 0:
                plt.scatter(subset[:, 0], subset[:, 1], color=color[idx], marker="d", s=13,alpha=0.8,label="Poisoned")
            subset = reduced_data[len(clean_labels)+int(0.5*len(poison_labels)):len(clean_labels)+len(poison_labels)][poison_labels.numpy()[int(0.5*len(poison_labels)):] == label]
            if len(subset) > 0:
                # print(len(subset))
                plt.scatter(subset[:, 0], subset[:, 1], color=color[idx], marker="d", s=13,alpha=0.8,label="Poisoned")
        plt.xlim((-60, 80))
        plt.ylim((-80, 80))
        # hfont = {'fontname':'Times New Roman'}
        plt.xlabel("Embedding 1")
        plt.ylabel("Embedding 2")
        plt.title("Embeddings of clean/poisoned images with respect based on PRN and AIN on CIFAR-10", loc='center', wrap=True)
        plt.legend(["clean","poisoned"] ,loc ="upper right",fontsize="8", frameon=False ,borderaxespad=0.2, # Legend closer to the border
            handletextpad=0.1,
            markerscale=1.5,    )
        plt.savefig("cifar_poisoned_defense_singledct.png")

        plt.savefig("cifar_poisoned_defense_singledct.pdf")
        plt.show()
    

    # input()
    #average dct data
    average_dct_data = torch.mean(data_dct, dim=0)
    plt.imshow(average_dct_data.detach().cpu().permute(1, 2, 0))
    plt.axis('off')

    # plt.show()

    average_dct_data_ps = torch.mean(data_dct_ps, dim=0)
    plt.imshow(average_dct_data_ps.detach().cpu().permute(1, 2, 0))
    plt.axis('off')

    print("backdoored model evaluation===================")

    evaluate(model, clean_testset, mode="Clean")
    evaluate(model, poisoned_testset, mode="Poisoned")

    print("defence model evaluation===================")

###using backdoor backdoor to defence
    #poisoned dct
    print("poisoned dct")
    evaluate_bb(model, clean_testset, average_dct_data_ps, mode="Clean")
    evaluate_bb(model, poisoned_testset,average_dct_data_ps, mode="Poisoned")
    #clean dct
    print("clean dct")
    evaluate_bb(model, clean_testset, average_dct_data, mode="Clean")
    evaluate_bb(model, poisoned_testset,average_dct_data, mode="Poisoned",use_clean_label=True,clean_testset=clean_testset)
    print("===================ASR===================")
    evaluate_bb(model, poisoned_testset,average_dct_data, mode="Poisoned",use_clean_label=False,clean_testset=clean_testset,exclude_target=True,target_label=0)


    print("defence model evaluation using single image===================")

    #posioned dct
    print("poisoned dct")
    evaluate_bb(model, clean_testset, three_channel_dct_one_image(poisoned_testset[1][0]).cuda(), mode="Clean")
    evaluate_bb(model, poisoned_testset, three_channel_dct_one_image(poisoned_testset[1][0]).cuda(), mode="Poisoned")

    #clean dct
    # plt.imshow(clean_testset[1][0].permute(1, 2, 0))
    # plt.axis('off')
    # plt.show()
    print("clean dct")
    # print(clean_testset[5][1])
    evaluate_bb(model, clean_testset,  three_channel_dct_one_image(clean_testset[10][0]).cuda(), mode="Clean")
    evaluate_bb(model, poisoned_testset, three_channel_dct_one_image(clean_testset[10][0]).cuda(), mode="Poisoned",use_clean_label=True,clean_testset=clean_testset)
    print("===================ASR===================")
    evaluate_bb(model, poisoned_testset, three_channel_dct_one_image(clean_testset[10][0]).cuda(), mode="Poisoned",use_clean_label=False,clean_testset=clean_testset,exclude_target=True,target_label=0)
