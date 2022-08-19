import torch
import Torture
import torchvision
import numpy as np
from Torture.Models import resnet_3layer as resnet
from Torture.loss_function import soft_cross_entropy

BATCH_SIZE = 32
lamb_range = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
def onehot(ind):
    vector = np.zeros([num_classes])
    vector[ind] = 1
    return vector.astype(np.float32)

train_trans, test_trans = Torture.Models.transforms.cifar_transform()
trainset = torchvision.datasets.CIFAR10(root='./LargeData/cifar/',
                                            train=True,
                                            download=True,
                                            transform=train_trans,
                                            target_transform=onehot)
testset = torchvision.datasets.CIFAR10(root='./LargeData/cifar/',
                                           train=False,
                                           download=True,
                                           transform=test_trans,
                                           target_transform=onehot)
dataloader_train = torch.utils.data.DataLoader(
    trainset,
    # batch_size=FLAGS.batch_size,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2)
dataloader_test = torch.utils.data.DataLoader(
    testset,
    # batch_size=FLAGS.batch_size,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2)
# load model
model_set = resnet
CLASSIFIER = model_set.model_dict['resnet50']
num_classes = 100
num_worker = 2
classifier = CLASSIFIER(num_classes = num_classes)
classifier.load_state_dict(torch.load('./results/cifar100models/models/eopch199.ckpt'))
classifier.eval()

criterion = soft_cross_entropy

#
TEST_BATCH = 0
for i, data_batch in enumerate(dataloader_train):
    # get the inputs; data is a list of [inputs, labels]
    if(i == TEST_BATCH):
        for lamb in lamb_range:
            print('lamb = ', lamb)
            img_batch, label_batch = data_batch
            ##print(img_batch.shape,label_batch.shape,label_batch[0])
            label_pre_init = classifier(img_batch)
            softmax_out_init = torch.nn.functional.softmax(label_pre_init, dim=1)
            combo_img = torch.zeros(int(BATCH_SIZE*(BATCH_SIZE-1)/2),3,32,32)
            combo_label = torch.zeros(int(BATCH_SIZE*(BATCH_SIZE-1)/2),num_classes)
            combo_softmax_out_label = torch.zeros(int(BATCH_SIZE * (BATCH_SIZE - 1) / 2), num_classes)
            for ith in range(BATCH_SIZE):
                for jth in range(ith+1,BATCH_SIZE):
                    idx = int(((BATCH_SIZE - 1) + (BATCH_SIZE - ith)) * ith / 2 + jth - ith - 1)
                    # print(index)
                    combo_img[idx] = lamb * img_batch[ith] + (1 - lamb) * img_batch[jth]
                    combo_label[idx] = lamb * label_pre_init[ith] + (1 - lamb) * label_pre_init[jth]
                    combo_softmax_out_label[idx] = lamb * softmax_out_init[ith] + (1 - lamb) * softmax_out_init[jth]
                # print(ith)
            # a = 0.4*img_batch[0] + 0.6*img_batch[1]
            # b = 0.4*label_pre_init[0] + 0.6*label_pre_init[1]
            # a = torch.reshape(a,(1,3,32,32))
            # print(a)
            # torch.nn.functional.one_hot(combo_label[100].argmax(), num_classes=num_classes)
            label_pre = classifier(combo_img)
            combo_softmax_out_pre = torch.nn.functional.softmax(label_pre, dim=1)

            # loss = criterion(label_pre, combo_label)
            # print('loss = ', loss)
            similarity = torch.cosine_similarity(combo_softmax_out_pre, combo_softmax_out_label, dim=1)
            print('cos simi mean = ', torch.mean(similarity))
            print('cos simi var = ', torch.var(similarity))
            print('cos simi max = ', torch.max(similarity))
            print('cos simi min = ', torch.min(similarity))

            # print(combo_label[100])
            # print(label_pre[100])
            # print('...')
            # print(combo_label[101])
            # print(label_pre[101])
            # print('...')
            # print(combo_label[102])
            # print(label_pre[102])
            # print('...')
            # print(combo_label[103])
            # print(label_pre[103])

