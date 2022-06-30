
def lr_scheduler(epoch, lr):
        if (epoch == 10):
                lr = lr * 2
        elif (epoch == 20):
                lr = lr * 5
        elif (epoch == 75):
                lr = 1e-3
        elif (epoch == 105):
                lr = 1e-4
        else:
                lr = lr
        return lr        
        