import torch
import time


def check_spec():
    print("This program checks if your device is CUDA compatible!")
    time.sleep(3.0)
    while True:
        if torch.cuda.is_available() == True:
            print(f'\n__torch.cuda.is_available()___ == {torch.cuda.is_available()}:\
                \n \n[INFO]\
                \nThis means that device posesses CUDA compatible GPU')

        else:
            print(f'\n__torch.cuda.is_available()___ == {torch.cuda.is_available()}:\
                \n \n[WARNING]\
                \n This means that device does not posesses CUDA compatible GPU and hence the NeuralNetwork wil train\
                \non the CPU which is relatively Slow and time consuming(may likely take more than a week or some days)')

        time.sleep(3.0)
        x = str(input("\n press 'Q' and hit enter to exit: "))
        if x == 'q' or 'Q':
            break
    exit()


check_spec()
