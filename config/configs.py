import torch, os
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = 3
hidden_unit = 32
output_shape = 2
train_size = 0.80
img_size = 256
# num_workers = os.cpu_count()
# print(f"Number of CPU cores available: {num_workers}")

# ========== MODEL PARAMETERS ==========
num_epochs = 50
batch_size = 64
learning_rate = 0.1
scheduler_activate = False #SA means Scheduler Activate
lr_decay_steps = 10
gamma = 0.5
gamma_value_increase_rate = None
stop_gamma_value_increase_epoch = None

# ========== REGULARIZATION PARAMETERS ==========
dropout_rate = None
l2_lambda = 0.0002

# ========= DATA TRANSFORMS =======
# Data transforms for training data
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    # transforms.RandomResizedCrop(244),
    # transforms.RandomCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=15),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
                      transforms.Resize((img_size, img_size)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


# ========== DATA PATHS ==========
train_path = './data/train'
test_path = './data/test'
evl_csv_file_name = 'test_run.csv'
config_file_path = './config/configs.py'

Loss_Curves = './storage/Loss_Curves'
