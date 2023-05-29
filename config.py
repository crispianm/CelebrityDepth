# config.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FACE_TRAIN_DIR = "./celeba/img_align_celeba/"
DEPTH_TRAIN_DIR = "./celeba/img_depth_celeba/"
FACE_VAL_DIR = "./celeba/align_val"
DEPTH_VAL_DIR = "./celeba/depth_val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 1
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 5
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "./checkpoints/disc_0.pth.tar"
CHECKPOINT_GEN = "./checkpoints/gen_0.pth.tar"
