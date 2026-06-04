import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as pyplot
from sklearn import metrics
from tqdm import tqdm
import seaborn as sns
import pandas as pd

# Author: Patrick (100599029)

# Defines the exact blueprint of the network so PyTorch knows where to put the saved weights.
class MobileNetClassificationNetwork(nn.Module):
    def __init__(self, total_unique_identities):
        super(MobileNetClassificationNetwork, self).__init__()

        self.base_feature_extractor = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        for network_parameter in self.base_feature_extractor.parameters():
            network_parameter.requires_grad = False

        number_of_input_features = self.base_feature_extractor.classifier[1].in_features

        self.intermediate_fully_connected_layer = nn.Linear(
            in_features=number_of_input_features, 
            out_features=1024,
            bias=True
        )
        
        self.intermediate_activation_function = nn.ReLU()

        self.dropout_regularisation_layer = nn.Dropout(
            p=0.5, 
            inplace=False
        )

        self.final_classification_layer = nn.Linear(
            in_features=1024, 
            out_features=total_unique_identities, 
            bias=True
        )

        self.base_feature_extractor.classifier = nn.Sequential(
            self.intermediate_fully_connected_layer,
            self.intermediate_activation_function,
            self.dropout_regularisation_layer,
            self.final_classification_layer
        )

    def forward(self, input_image_batch):
        final_prediction_output = self.base_feature_extractor(input_image_batch)
        return final_prediction_output

# Ensures the code executes safely as the main program.
if __name__ == '__main__':

    # Defines the simple relative paths required to locate the data from the root folder.
    validation_text_file_path = "datasets/verification_pairs_val.txt"
    trained_model_weights_path = "models/verification_classification_patrick.pt"
    datasets_base_folder = "datasets/"

    # Defines the original output size of the training phase.
    total_training_identities = 4000

    # Builds the empty network architecture in memory.
    face_verification_model = MobileNetClassificationNetwork(total_unique_identities=total_training_identities)

    # Pours the saved mathematical weights into the empty architecture.
    model_weights = torch.load(trained_model_weights_path)
    face_verification_model.load_state_dict(model_weights)

    # Strips off the final guessing layer to expose the 1024-number embedding.
    face_verification_model.base_feature_extractor.classifier[3] = nn.Identity()

    # Disables training mode to ensure consistent evaluation behaviour.
    face_verification_model.eval()

    # Moves the model to the graphics card to optimise processing speed.
    graphics_card_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_verification_model.to(graphics_card_device)

    # Matches the image transformations perfectly to the training phase format.
    image_transformation_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialises empty lists to store the extracted numbers for the final graph calculation.
    euclidean_distance_scores_list = []
    cosine_similarity_scores_list = []
    ground_truth_labels_list = []

    # Opens the text file and reads all the verification pairs into memory.
    with open(validation_text_file_path, "r") as validation_file_handle:
        all_validation_lines = validation_file_handle.readlines()

    # Creates a visual progress bar in the terminal to monitor the evaluation speed.
    progress_bar = tqdm(all_validation_lines, desc="Evaluating Image Pairs")

    # Loops through every single line in the text file using the progress bar.
    for current_line in progress_bar:
        
        # Cleans the text and splits the line into three usable elements.
        cleaned_line_string = current_line.strip()
        split_line_elements = cleaned_line_string.split(" ")
        
        # Assigns the file paths and the correct answer label to variables.
        first_image_relative_path = split_line_elements[0]
        second_image_relative_path = split_line_elements[1]
        correct_label_integer = int(split_line_elements[2])
        
        # Adds the datasets prefix to build the complete file path.
        first_image_full_path = datasets_base_folder + first_image_relative_path
        second_image_full_path = datasets_base_folder + second_image_relative_path
        
        # Loads the physical image files and standardises the colour format.
        first_image_file = Image.open(first_image_full_path).convert('RGB')
        second_image_file = Image.open(second_image_full_path).convert('RGB')
        
        # Applies the mathematical transformations and moves the images to the graphics card.
        first_image_tensor = image_transformation_pipeline(first_image_file).unsqueeze(0).to(graphics_card_device)
        second_image_tensor = image_transformation_pipeline(second_image_file).unsqueeze(0).to(graphics_card_device)
        
        # Disables gradient tracking to drastically reduce memory usage.
        with torch.no_grad():
            
            # Passes the images through the network to generate the embeddings.
            first_face_embedding = face_verification_model(first_image_tensor)
            second_face_embedding = face_verification_model(second_image_tensor)
            
            # Calculates the Euclidean distance and converts it into a similarity score.
            euclidean_distance = torch.nn.functional.pairwise_distance(first_face_embedding, second_face_embedding)
            euclidean_similarity_score = 1.0 / (1.0 + euclidean_distance)
            extracted_euclidean_number = euclidean_similarity_score.item()
            
            # Calculates the Cosine similarity directly.
            cosine_similarity_score = torch.nn.functional.cosine_similarity(first_face_embedding, second_face_embedding)
            extracted_cosine_number = cosine_similarity_score.item()
            
            # Appends the final numbers to the tracking lists.
            euclidean_distance_scores_list.append(extracted_euclidean_number)
            cosine_similarity_scores_list.append(extracted_cosine_number)
            ground_truth_labels_list.append(correct_label_integer)

    # Calculates the final metrics required to draw the ROC curves.
    euclidean_false_positive_rate, euclidean_true_positive_rate, _ = metrics.roc_curve(ground_truth_labels_list, euclidean_distance_scores_list)
    euclidean_auc_value = metrics.auc(euclidean_false_positive_rate, euclidean_true_positive_rate)

    cosine_false_positive_rate, cosine_true_positive_rate, _ = metrics.roc_curve(ground_truth_labels_list, cosine_similarity_scores_list)
    cosine_auc_value = metrics.auc(cosine_false_positive_rate, cosine_true_positive_rate)

    # Creates the physical graph layout.
    pyplot.figure(figsize=(10, 8))
    pyplot.plot(euclidean_false_positive_rate, euclidean_true_positive_rate, label=f"Euclidean Distance AUC: {euclidean_auc_value:.4f}")
    pyplot.plot(cosine_false_positive_rate, cosine_true_positive_rate, label=f"Cosine Similarity AUC: {cosine_auc_value:.4f}")
    pyplot.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess Baseline (AUC: 0.5000)")

    # Applies all the necessary labels and titles to the graph.
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("System Evaluation: ROC Curve for Face Verification Metrics")
    pyplot.legend()

    # Physically saves the final graph as an image file in the reports folder.
    final_image_filename = "reports/Patrick_MobileNetV2_ROC_Curve.png"
    pyplot.savefig(final_image_filename, bbox_inches='tight')
    
    # Prints a final confirmation message to the terminal.
    print(f"\nEvaluation complete. Graph successfully saved to: {final_image_filename}")

    print("Generating Score Distribution Density Plot...")

    # Separate the cosine scores based on whether they were actual matches (1) or imposters (0)
    genuine_scores = [score for score, label in zip(cosine_similarity_scores_list, ground_truth_labels_list) if label == 1]
    imposter_scores = [score for score, label in zip(cosine_similarity_scores_list, ground_truth_labels_list) if label == 0]

    # Create a clean, academic plot
    pyplot.figure(figsize=(10, 6))
    sns.kdeplot(genuine_scores, fill=True, color="blue", label="Genuine Pairs (Same Person)", alpha=0.5)
    sns.kdeplot(imposter_scores, fill=True, color="red", label="Imposter Pairs (Different People)", alpha=0.5)

    pyplot.title("Cosine Similarity Score Distribution (MobileNetV2)")
    pyplot.xlabel("Cosine Similarity Score")
    pyplot.ylabel("Density")
    pyplot.legend()
    pyplot.grid(True, linestyle='--', alpha=0.7)

    # Save the graph
    distribution_filename = "reports/Patrick_MobileNetV2_Score_Distribution.png"
    pyplot.savefig(distribution_filename, bbox_inches='tight')
    print(f"Distribution graph saved to: {distribution_filename}")