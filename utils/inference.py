from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from config import configs
import utils
import numpy as np
import pandas as pd
import torch, csv
import os, random
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score,
                             confusion_matrix, average_precision_score, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from model import LightCDC
from tqdm import tqdm



def get_class_name_from_index(index, test_path):
    # Ensure the classes are always listed in the same order by sorting
    classes = sorted(os.listdir(test_path))
    # Return the class name that corresponds to the index
    return classes[index]


def single_image_inference(model_weight, image_path, test_path = configs.test_path, transform = configs.val_test_transform, device = configs.device):

    model = LightCDC()
    state_dict = torch.load(model_weight, map_location=configs.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(configs.device)
    model.eval()

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to the device

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Retrieve the highest probability
    predicted_probability, _ = predicted_probabilities.max(1)

    # Extract the actual class name from the image path
    actual_class_name = os.path.basename(os.path.dirname(image_path))

    # Use the predicted index to get the class name
    predicted_class_name = get_class_name_from_index(predicted_idx, test_path)

    # Print the results
    print(f"Actual class: {actual_class_name}")
    print(f"Predicted class: {predicted_class_name}")
    print(f"Predicted probability: {predicted_probability.item()}")

    return actual_class_name, predicted_class_name, predicted_probability.item()


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def multiple_inference(model_name="lightCDC", model_weight=None, evl_csv_file_name="test_result", batch_size=None,
                       lr=None, test_folder=None):
    # === Auto-create output subfolder (test_1, test_2, ...) ===
    base_dir = configs.evaluation_output_root if hasattr(configs, 'evaluation_output_root') else './evaluation_results'
    os.makedirs(base_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(base_dir) if
                     d.startswith('test_') and os.path.isdir(os.path.join(base_dir, d))]
    run_ids = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
    next_run_id = max(run_ids, default=0) + 1
    run_folder = os.path.join(base_dir, f'test_{next_run_id}')
    os.makedirs(run_folder, exist_ok=True)

    # Define output paths
    auc_pr_path = os.path.join(run_folder, f"{model_name}_Multiclass_AUC_PR.jpg")
    confusion_matrix_path = os.path.join(run_folder, "confusion_matrix.jpg")
    csv_file_path = os.path.join(run_folder, f"{evl_csv_file_name}.csv")
    metrics_path = os.path.join(run_folder, f"{model_name}_CDC_Testset.txt")

    # === Load model and test data ===
    test_loader, class_to_idx = utils.test_loader_function(batch_size=batch_size, test_path=test_folder)
    learning_rate = lr

    model = LightCDC()
    model.load_state_dict(torch.load(model_weight, map_location=configs.device))
    model.to(configs.device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    correct_top2_count = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", unit="batch"):
            data, target = data.to(configs.device), target.to(configs.device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

            _, top2_preds = outputs.topk(2, 1, True, True)
            correct_top2_count += torch.eq(top2_preds, target.unsqueeze(dim=1)).sum().item()

    all_probs = np.array(all_probs)
    n_samples = len(all_labels)
    top1_accuracy = accuracy_score(all_labels, all_preds)
    top2_accuracy = correct_top2_count / n_samples

    n_classes = configs.output_shape
    all_labels_binarized = label_binarize(all_labels, classes=range(n_classes))

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    positive_class_probs = all_probs[:, 1]
    pr_auc = average_precision_score(all_labels_binarized, positive_class_probs, average="micro")

    print('Evaluation on CDC Testset:')
    print(
        f'Model: {model_name} | Top-1 Acc: {round(top1_accuracy * 100, 2)} | Precision: {round(precision * 100, 2)} | Recall: {round(recall * 100, 2)} | F1-score: {round(f1_score * 100, 2)}')

    # === Save metrics to CSV ===
    new_data_row = {
        'Model': model_name,
        'Epochs': configs.num_epochs,
        'BS': batch_size,
        'LR': learning_rate,
        'Scheduler': configs.scheduler_activate,
        'Acc': round(top1_accuracy * 100, 2),
        'Precision': round(precision * 100, 2),
        'Recall': round(recall * 100, 2),
        'F1-score': round(f1_score * 100, 2),
        'MCC': round(mcc * 100, 2),
        'Kappa': round(kappa * 100, 2),
        'AUC-PR': round(pr_auc * 100, 2)
    }

    pd.DataFrame([new_data_row]).to_csv(csv_file_path, index=False)

    # === Save Precision-Recall Curve ===
    plt.figure()
    if n_classes == 2:
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels_binarized, all_probs[:, 1])
        average_precision = average_precision_score(all_labels_binarized, all_probs[:, 1])
        plt.plot(recall_curve, precision_curve, lw=2, label=f'Class 1 PR (area = {average_precision:.2f})')
    else:
        for i in range(n_classes):
            precision_curve, recall_curve, _ = precision_recall_curve(all_labels_binarized[:, i], all_probs[:, i])
            average_precision = average_precision_score(all_labels_binarized[:, i], all_probs[:, i])
            plt.plot(recall_curve, precision_curve, lw=2, label=f'Class {i} PR (area = {average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(auc_pr_path)
    plt.close()

    # === Save Confusion Matrix ===
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=[idx_to_class[i] for i in range(n_classes)],
                yticklabels=[idx_to_class[i] for i in range(n_classes)])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(confusion_matrix_path)
    plt.close()

    print(f"Results saved to: {run_folder}")
