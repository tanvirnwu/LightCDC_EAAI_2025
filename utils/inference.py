from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from config.configs import *
import utils
import numpy as np
import pandas as pd
import torch, csv
import os, random
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score,
                             confusion_matrix, average_precision_score, precision_recall_curve)
from sklearn.preprocessing import label_binarize


def single_random_inference(model, model_name, hazy_type):
    # Data preparation now also returns class_to_idx mapping
    test_loader, class_to_idx = utils.test_loader_function()
    model_path = os.path.join(r'F:\Research\CDC5k\Storage\Saved_Models\\', f'{model_name}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Switch model to evaluation mode
    model.eval()

    # Filter test_loader to only include data of the specified hazy_type
    filtered_test_samples = [(image, label) for image, label in test_loader.dataset if class_to_idx[hazy_type] == label]

    if not filtered_test_samples:
        raise ValueError(f"No images found for class '{hazy_type}'")

    # Randomly select one image and label of the specified hazy_type
    image, true_label = random.choice(filtered_test_samples)

    # Display the image
    utils.display_image(image)  # Ensure you have a function to display the image.

    # Move the image to the device and add a batch dimension
    image = image.to(device).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_prob, predicted_label = torch.max(probabilities, 1)

    # Convert to actual class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    actual_class_name = idx_to_class[true_label]
    predicted_class_name = idx_to_class[predicted_label.item()]

    # Print the actual and predicted class along with the prediction probability
    print(f'Actual class: {actual_class_name}')
    print(f'Predicted class: {predicted_class_name}')
    print(f'Prediction probability: {predicted_prob.item():.4f}')


def get_class_name_from_index(index, test_path):
    # Ensure the classes are always listed in the same order by sorting
    classes = sorted(os.listdir(test_path))
    # Return the class name that corresponds to the index
    return classes[index]


def single_image_inference(model, model_name, image_path, test_path = test_path, transform = val_test_transform, device = device):
    # Load the model
    model_path = os.path.join(r'D:\Research\CDC5k\Storage\Saved_Models', f'{model_name}.pth')
    # model_path = r"D:\Research\CDC5k\Storage\Saved_Models\ShuffleNetV2_Custom_CDC_ES10_ADAM_50_64_le-3.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
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




# Set random seeds and deterministic behavior for consistency
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def multiple_inference(model = None, model_name = None, evl_csv_file_name = None, batch_size = None, lr = None):
    test_loader, class_to_idx = utils.test_loader_function(batch_size= batch_size)
    learning_rate = lr

    # =========Load the model========
    if model != None:
        model_path = os.path.join(r'F:\Research\CDC5k\Storage\Saved_Models', f'{model_name}.pth')
        # print(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    correct_top2_count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if model_name == 'Bagging1':
                outputs = model.Bagging1(data)
            elif model_name == 'Bagging2':
                outputs = model.Bagging2(data)
            elif model_name == 'Bagging3':
                # print('Inside Inference: Bagging3')
                outputs = model.Bagging3(data)
            elif model_name == 'Bagging4':
                outputs = model.Bagging4(data)
            elif model_name == 'Stacking1':
                outputs = model.Stacking1(data)
            elif model_name == 'Stacking2':
                outputs = model.Stacking2(data)
            elif model_name == 'Stacking3':
                outputs = model.Stacking3(data)
            elif model_name == 'Stacking4':
                outputs = model.Stacking4(data)
            elif model_name == 'Stacking5':
                outputs = model.Stacking5(data)

            elif model_name == 'Voting1':
                outputs = model.Voting1(data)
            elif model_name == 'Voting2':
                outputs = model.Voting2(data)
            elif model_name == 'Voting3':
                outputs = model.Voting3(data)
            elif model_name == 'Voting4':
                outputs = model.Voting4(data)
            elif model_name == 'Voting5':
                outputs = model.Voting5(data)
            else:
                outputs = model(data)

            # outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

            # Calculating top-2 accuracy
            _, top2_preds = outputs.topk(2,1, True, True)
            correct_top2_count += torch.eq(top2_preds, target.unsqueeze(dim=1)).sum().item()


    # Convert all_probs to a NumPy array for further processing
    all_probs = np.array(all_probs)
    n_samples = len(all_labels)
    top1_accuracy = accuracy_score(all_labels, all_preds)
    top2_accuracy = correct_top2_count / n_samples

    n_classes = Utils.output_shape
    all_labels_binarized = label_binarize(all_labels, classes=range(n_classes))

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    # print(all_labels_binarized.shape)  # Should match the number of samples and the number of classes
    # print(all_probs.shape)
    positive_class_probs = all_probs[:, 1]
    pr_auc = average_precision_score(all_labels_binarized, positive_class_probs, average="micro")

    print('Evaluation on CIFAKE Testset:')
    print(f'Model: {model_name} | Top-1 Acc: {round(top1_accuracy*100,2)} | Precision: {round(precision*100,2)} | Recall: {round(recall*100,2)} | F1-score: {round(f1_score*100,2)}')

    metrics_path = os.path.join(evaluation_metrics_testset, f'{model_name}_CIFAKE_Testset.txt')
    auc_pr_save = auc_pr_save_testset
    confusion_matrix_save_path = os.path.join(confusion_matrix_save_testset, f'{model_name}_CIFAKE_Testset.jpg')

    # Ensure the directory exists, create it if it does not
    os.makedirs(evaluation_metrics_csv_dir, exist_ok=True)

    # Define the full path to the CSV file
    csv_file_path = os.path.join(evaluation_metrics_csv_dir, f'{evl_csv_file_name}.csv')

    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)
    if file_exists:
        existing_data = pd.read_csv(csv_file_path)
    else:
        existing_data = pd.DataFrame()

    new_data_row = {
        'Model': model_name,
        'Epochs': num_epochs,
        'BS': batch_size,
        'LR': learning_rate,
        'Scheduler': scheduler_activate,
        'Acc': round(top1_accuracy*100, 2),
        'Precision': round(precision*100, 2),
        'Recall': round(recall*100, 2),
        'F1-score': round(f1_score*100, 2),
        'MCC': round(mcc*100, 2),
        'Kappa': round(kappa*100, 2),
        'AUC-PR': round(pr_auc*100, 2)
    }

    # Convert the new data row to a DataFrame to use pandas functionality
    new_data_df = pd.DataFrame([new_data_row])

    # Concatenate the new data with existing data and check for duplicates
    combined_data = pd.concat([existing_data, new_data_df], ignore_index=True)
    if not combined_data.duplicated().iloc[-1]:
        # Writing data to CSV file (only if it's not a duplicate)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=new_data_row.keys())
            # Write the header only if the file is being created for the first time
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_data_row)

    # Save and Plot Precision-Recall curves for multiclass classification
    plt.figure()

    # Calculate and plot PR curve for each class
    for i in range(n_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels_binarized, all_probs[:, i])
        average_precision = average_precision_score(all_labels_binarized, all_probs[:, i])
        plt.plot(recall_curve, precision_curve, lw=2, label=f'Class {i} PR (area = {average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(auc_pr_save, f'{model_name}_Multiclass_AUC_PR.jpg'))
    # plt.show()
    # plt.close()

    # Invert the class_to_idx dictionary to get a mapping from index to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Assuming conf_matrix, n_classes, idx_to_class, and model_name are defined
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                     xticklabels=[idx_to_class[i] for i in range(n_classes)],
                     yticklabels=[idx_to_class[i] for i in range(n_classes)])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xticks(rotation=45)  # Adjusted for better readability if labels overlap
    plt.yticks(rotation=45)  # Adjusted for better readability if labels overlap
    # Uncomment the next line to save the plot
    plt.savefig(confusion_matrix_save_path)
    # plt.show()
    plt.close()


