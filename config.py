class Train_config:
    def __init__(self):
        self.train_data_dir   = "data/train_preprocessed.xlsx"
        self.test_data_dir    = "data/test_preprocessed.xlsx"
        self.test_fill_dir    = "data/test.xlsx"
        self.save_dir         = "data/model/NN.pt"
        self.device           = "cuda"
        self.max_epoch        = 50
        self.lr               = 0.001
        self.train_batch_size = 32
        self.test_batch_size  = 200
