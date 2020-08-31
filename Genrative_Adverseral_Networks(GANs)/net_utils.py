import numpy as np
from tqdm import tqdm
import os, cv2, torch
import torch.nn as nn
from torchvision.utils import save_image
import random
import matplotlib.pyplot as plt
import time


#Data preprocessing
data = []
def dataPreprocessor(raw_data_path, save_data_path):

    for single_data in tqdm(os.listdir(raw_data_path)):

        try:
            img = os.path.join(raw_data_path, single_data)
            img = cv2.imread(img)
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.array(img, dtype=np.uint8)
            data.append([img])

        except Exception as e:
            pass
   
    np.random.shuffle(data)
    np.save(save_data_path, data, allow_pickle=True)

device = torch.device('cuda' if torch.cuda.is_available() == True else 'cpu')

raw_data_path = 'raw_data/training_data'
train_path = 'processed_data/training_dataset/train_dataset.npy'
saved_model_path = 'saved_model/DCGAN_network.pth.tar'
#Hyper-parameters
in_channels_D = 3
out_channels_D = 1
in_noise_channels_G = 256
out_channels_G = in_channels_D
lr = 0.0002
EPOCHS = 1000
fake_label = 1 #~0
real_label = 0 #~1
num_of_eval_data = 3
desciminator_loss = []
Generator_loss = []

def save_model(model, path=saved_model_path):
    print('Saving model.......')
    torch.save(model, path)
    print('AI model has been saved.....')


def trainData():
    if not os.path.isfile(train_path):
        print("Preprocessing Data....... ")
        dataPreprocessor(raw_data_path=raw_data_path, save_data_path=train_path)
    else:
        pass


trainData()


from neuralNets import Descriminator, Generator
DescNet = Descriminator(in_channels_D, out_channels_D).to(device=device)
GenNet = Generator(in_noise_channels_G, out_channels_G).to(device=device)


import torch.optim as optim
optimizer_D = optim.Adam(DescNet.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_G = optim.Adam(GenNet.parameters(), lr=lr, betas=(0.5, 0.999))
lossFunc = nn.BCELoss()


def dataLoader(path):
    data = np.load(path, allow_pickle=True)
    X = np.array([i[0] for i in data])
    X = X / 255
    X = X.reshape(len(X), 64, 64, -1)
    X = X.transpose(0, 3, 1, 2)
    np.random.shuffle(X)
    X = torch.Tensor(X)
    return X


def batch_shuffler(tensor):
    tensor = np.array(tensor.detach())
    np.random.shuffle(tensor)
    tensor = torch.Tensor(tensor)
    return tensor


def trainProcess(epochs, BATCH_SIZE = 16):

    if not os.path.isfile(saved_model_path):

        epoch_tab = []
        Gen_loss = np.array([])
        desc_loss = np.array([])


        for epoch in range(epochs):
            X = batch_shuffler(tensor=dataLoader(path=train_path))
            epoch_tab.append(epochs)
            for imgs in tqdm(range(0, len(X), BATCH_SIZE)):
                X_batch = X[imgs:imgs+BATCH_SIZE]
                noise = torch.randn(X_batch.shape[0], in_noise_channels_G, 1, 1).to(device=device)
                real_target = torch.Tensor(1, X_batch.shape[0]).fill_(real_label).to(device=device)
                fake_target = torch.Tensor(1, X_batch.shape[0]).fill_(fake_label).to(device=device)

                #Train Generator
                GenNet.zero_grad()
                G_output = GenNet(noise)
                D_output = DescNet(G_output)
                G_loss = lossFunc(D_output.reshape(1, -1), real_target)
                G_loss.backward()
                optimizer_G.step()

                #Train Descrimintor
                #Train Descriminator on real dataset
                DescNet.zero_grad()
                real_output = DescNet(X_batch)
                real_loss = lossFunc(real_output.reshape(1, -1), real_target*0.1)

                #Train Descriminator on fake data
                r'the gradient of the output of the generator is detached,\
                while training the Descriminator to prevent back_prop on it'
                fake_output = DescNet(G_output.detach())
                fake_loss = lossFunc(fake_output.reshape(1, -1), fake_target*0.9)

                D_loss = (real_loss + fake_loss)
                D_loss.backward()
                optimizer_D.step()
                
                #append losses
                desc_loss = np.append(desc_loss, D_loss.data)
                Gen_loss = np.append(Gen_loss, G_loss.data)

                if epoch % 2 == 0 and X_batch.shape[0] == 16:
                    out = G_output*255
                    out = nn.functional.interpolate(out, mode='nearest', scale_factor=2)
                    save_image(out[random.randrange(0, 15)], f'output/{epoch}.jpg', nrow=1, normalize=True)

            desciminator_loss.append(np.mean(desc_loss))
            Generator_loss.append(np.mean(Gen_loss))
            print(f'epoch: {epoch+1}\t desc_loss: {np.mean(desc_loss)}\t Gen_loss: {np.mean(Gen_loss)}\n')
            D_real_loss = np.array([])
            D_fake_loss = np.array([])
            Gen_loss = np.array([])

        model_state = {'Gen_model':GenNet.state_dict(), 'Gen_optimizer':optimizer_D.state_dict(),
                        'Desc_model':DescNet.state_dict(),'Desc_optimizer':optimizer_D.state_dict(),
                        'loss_func':lossFunc.state_dict()}
        save_model(model_state)

    else:
        pass


def evalProcess(no_img):
    fixed_input_noise = torch.randn(no_img, in_noise_channels_G, 1, 1).to(device=device)
    with torch.no_grad():
        model_data = torch.load(saved_model_path)
        Gen_model = Generator(in_noise_channels_G, out_channels_G)
        Gen_model.load_state_dict(model_data['Gen_model'])
        Gen_model.eval().to(device=device)
        for num, img in zip(range(no_img), fixed_input_noise):
            img = img.to(device=device)
            img = img.view(1, in_noise_channels_G, 1, 1)
            output_img = Gen_model(img)
            #reverse data normalization
            output_img = np.array(output_img*255, dtype=np.uint8)
            #N x C x H x W
            output_img = output_img.transpose(0, 2, 3, 1)
            #output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)
            cv2.imshow(f'data{num}', output_img[0])
            time.sleep(3)
        cv2.waitKey(0)
        

trainProcess(EPOCHS)
evalProcess(num_of_eval_data)












