from parameters import *
import torch as t
from torch import optim
from torch import nn
from dataset import *
from torch.utils.data import DataLoader
import tqdm
from tensorboardX import SummaryWriter
from torchnet import meter


writer = SummaryWriter('./tensorboard/' + 'resnet')


def train(model):
    avgLoss = 0.0
    best_acc = 0.0
    save_path = './weights/captcha'
    os.makedirs(save_path, exist_ok=True)
    if t.cuda.is_available():
        model = model.cuda()
    # data loading
    trainDataset = Captcha("../captcha/train/", train=True)
    testDataset = Captcha("../captcha/test/", train=False)
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
                                 shuffle=True, num_workers=4)
    testDataLoader = DataLoader(testDataset, batch_size=batchSize,
                                shuffle=True, num_workers=4)
    circles_per_epoch = len(trainDataLoader) // batchSize
    # max_iters = circles_per_epoch * circles_per_epoch
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    loss_meter = meter.AverageValueMeter()
    # training procedure
    for epoch in range(totalEpoch):
        for circle, input in tqdm.tqdm(enumerate(trainDataLoader, 0)):
            x, label = input
            if t.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2, label3, label4 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
            # print(label1,label2,label3,label4)
            optimizer.zero_grad()
            y1, y2, y3, y4 = model(x)
            # print(y1.shape, y2.shape, y3.shape, y4.shape)
            loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(y2, label2) \
                , criterion(y3, label3), criterion(y4, label4)
            loss = loss1 + loss2 + loss3 + loss4
            loss_meter.add(loss.item())
            writer.add_scalar('train/loss', loss.item(), circle + epoch * circles_per_epoch)
            # print(loss)
            avgLoss += loss.item()
            loss.backward()
            optimizer.step()
            # evaluation
            if circle % printCircle == 1:
                print("Epoch %d : after %d circle,the train loss is %.5f" %
                      (epoch, circle, avgLoss / printCircle))
                writeFile("Epoch %d : after %d circle,the train loss is %.5f" %
                          (epoch, circle, avgLoss / printCircle))

                avgLoss = 0
            if circle % testCircle == 1:
                accuracy = test(model, testDataLoader)
                if accuracy > best_acc:
                    best_acc = accuracy
                    model.save(save_path)
                print('current acc is : {}, the best acc is : {}'.format(accuracy, best_acc))
                writeFile("current acc is : %.5f, the best acc is : %.5f" % (accuracy, best_acc))
                writer.add_scalar('test/acc', accuracy, circle + epoch * circles_per_epoch)
            # if circle % saveCircle == 1:
            #     model.save(str(epoch)+"_"+str(saveCircle))
    writer.close()


def test(model, testDataLoader):
    totalNum = testNum * batchSize
    rightNum = 0
    for circle, input in enumerate(testDataLoader, 0):
        if circle >= testNum:
            break
        x, label = input
        label = label.long()
        if t.cuda.is_available():
            x = x.cuda()
            label = label.cuda()
        y1, y2, y3, y4 = model(x)
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(batchSize, 1), y2.topk(1, dim=1)[1].view(batchSize, 1), \
                         y3.topk(1, dim=1)[1].view(batchSize, 1), y4.topk(1, dim=1)[1].view(batchSize, 1)
        y = t.cat((y1, y2, y3, y4), dim=1)
        diff = (y != label)
        diff = diff.sum(1)
        diff = (diff != 0)
        res = diff.sum(0).item()
        rightNum += (batchSize - res)
    # print("the accuracy of test set is %s" % (str(float(rightNum) / float(totalNum))))
    # writeFile("the accuracy of test set is %s" % (str(float(rightNum) / float(totalNum))))
    return float(rightNum) / float(totalNum)


def writeFile(str):
    file = open("result_100w.txt", "a+")
    file.write(str)
    file.write("\n\n")
    file.flush()
    file.close()
