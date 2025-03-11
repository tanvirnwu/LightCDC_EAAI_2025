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
# batch_size = 64
# learning_rate = 0.1
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
train_path = r'F:\Research\CDC5k\Data\Train'
# test_path = r'F:\Research\XHC\Data\RH Testset\RH Test 3'

test_path = r'D:\Research\CDC5k\Data\Test'

evl_csv_file_name = 'AblationStudy.csv'

config_file_path = r'F:\Research\CDC5k\Utils\Config.py'
# Path where you want to save the new text file
# save_file_path = r'F:\Research\CDC5k\Storage\Save_Configs\\' + model_name + '_Configs.txt'

# ========= STORAGE PATHS =========
evaluation_metrics_csv_dir = r'F:\Research\CDC5k\Storage'
auc_pr_save_testset = r"F:\Research\CDC5k\Storage\AUC_PR"
evaluation_metrics_testset = r'F:\Research\CDC5k\Storage\Evaluation_Metrics'
confusion_matrix_save_testset = r'F:\Research\CDC5k\Storage\Confusion_Matrix'
Loss_Curves = r'F:\Research\CDC5k\Storage\Loss_Curves'
# model_weights_saving_path = r'F:\Research\CDC5k\Storage\Saved_Models\\' + model_name + '.pth'
