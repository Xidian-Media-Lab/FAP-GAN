class args():

    # hyperparameters
    put_type = 'right'
    balance = 1

    # training args
    epochs = 100# "number of training epochs, default is 2"
    trainpath = "./trainimage/"
    save_per_epoch = 1
    # trainG1_continue = "off"
    batch_size = 8# "batch size for training/testing, default is 4"
    # HEIGHT = 1024
    # WIDTH = 672
    lr = 1e-4 # "Initial learning rate, default is 0.0001"
    lr_step = 10 # Learning rate is halved in 10 epochs
    #SSIM_resume = "./models/best.SSIM.ckpt_2.pt" # if you have, please put the path of the model like "./models/densefuse_gray.model"
    #tv_resume = "./models/best.TV.ckpt_2.pt"
    resume = None
    #resume = "./models/Wrap.ckpt_5.pt"
    #resume = "/media/lab/liulu/liulu/LRINet_Copy/models/fakegt28_0.5_1.5_150.pt"

    save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."
    workers = 0
    #train_continue = "on"
    pre_train = "off"
    train_continue = "on"
    st_eopch = 0
    mode = "train"
    dir_chck = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/multiresolutionGFVC/checkpoints/"
    dir_log_train = "./log/"

    # For GPU training
    world_size = -1
    rank = -1
    gpu = 0
    multiprocessing_distributed = False
    distributed = None


