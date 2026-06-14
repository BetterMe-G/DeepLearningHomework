"""
Global configuration for DCGAN face-generation project.
Change values here or override from the command line in each script.
"""
import os

class Config:
    # --------- Data ---------
    dataset = "celeba"             # 'lfw' (default, from Kaggle atulanandjha/lfwpeople) or 'celeba' or 'folder'
    data_root = "./data/celeba/img_align_celeba"    # root that contains the images (e.g. ./data/lfw/lfw-deepfunneled)
    image_size = 64             # spatial size of training images
    channels = 3                # RGB

    # --------- Model ---------
    z_dim = 100                 # latent vector size
    g_feat = 64                 # base feature multiplier in Generator
    d_feat = 64                 # base feature multiplier in Discriminator

    # --------- Training ---------
    batch_size = 128
    num_workers = 4
    epochs = 80
    lr = 2e-4                   # base lr (used for G)
    lr_d = 2e-4                 # keep D/G pace matched for faster early convergence
    beta1 = 0.5                 # Adam beta1, classic DCGAN value
    beta2 = 0.999
    label_smooth = 1.0          # keep targets sharp by default
    d_noise = 0.0               # disable instance noise by default
    hflip_p = 0.0               # disable augmentation by default
    ema_decay = 0.0             # disable EMA by default (turn on after baseline is stable)
    use_ema_for_eval = True     # prefer EMA weights in generate/evaluate/interpolate

    # --------- Logging / Saving ---------
    out_dir = "./checkpoints"
    sample_dir = "./samples"
    log_dir = "./logs"
    save_every = 1              # save checkpoint every N epochs
    sample_every = 200          # save sample image grid every N iterations
    log_every = 50              # print loss every N iterations

    # --------- Eval ---------
    fid_num_samples = 10000     # number of fake samples for FID
    fid_real_dir = "./data/fid_real"
    fid_fake_dir = "./samples/fid_fake"

    # --------- Misc ---------
    seed = 42
    device = "cuda"             # 'cuda' or 'cpu'

    @classmethod
    def ensure_dirs(cls):
        for p in [cls.out_dir, cls.sample_dir, cls.log_dir,
                  cls.fid_real_dir, cls.fid_fake_dir]:
            os.makedirs(p, exist_ok=True)
