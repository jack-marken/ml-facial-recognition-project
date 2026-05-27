import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as pyplot
from sklearn import metrics
from tqdm import tqdm

# This directly imports Zhongyu's model loading function from his module.
from face_verification.metric_learning.embedding_model_zhongyu import load_embedding_model

# Ensures the code executes safely as the main program.
if __name__ == '__main__':

    # Defines the relative paths for the dataset and Zhongyu's specific model file.
    validation_text_file_path = "datasets/verification_pairs_val.txt"
    trained_model_weights_path = "models/recognition_triplet_resnet34_zhongyu.pth"
    datasets_base_folder = "datasets/"

    # Uses Zhongyu's custom function to build his ResNet34 architecture and pour his weights in.
    face_verification_model = load_embedding_model(
        architecture="resnet34",
        model_path=trained_model_weights_path,
        pretrained=False 
    )

    # Moves the model to the graphics card to optimise processing speed.
    graphics_card_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_verification_model.to(graphics_card_device)

    # Matches the exact image transformations used in your test to ensure a fair comparison.
    image_transformation_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialises empty lists to store the extracted numbers.
    euclidean_distance_scores_list = []
    cosine_similarity_scores_list = []
    ground_truth_labels_list = []

    # Opens the text file and reads all the verification pairs into memory.
    with open(validation_text_file_path, "r") as validation_file_handle:
        all_validation_lines = validation_file_handle.readlines()

    # Creates a visual progress bar in the terminal.
    progress_bar = tqdm(all_validation_lines, desc="Evaluating Zhongyu's ResNet34")

    # Loops through every single line in the text file.
    for current_line in progress_bar:
        
        # Cleans the text and splits the line.
        cleaned_line_string = current_line.strip()
        split_line_elements = cleaned_line_string.split(" ")
        
        # Assigns the file paths and the correct answer label.
        first_image_relative_path = split_line_elements[0]
        second_image_relative_path = split_line_elements[1]
        correct_label_integer = int(split_line_elements[2])
        
        # Builds the complete file path.
        first_image_full_path = datasets_base_folder + first_image_relative_path
        second_image_full_path = datasets_base_folder + second_image_relative_path
        
        # Loads the physical image files and standardises the colour format.
        first_image_file = Image.open(first_image_full_path).convert('RGB')
        second_image_file = Image.open(second_image_full_path).convert('RGB')
        
        # Applies the mathematical transformations.
        first_image_tensor = image_transformation_pipeline(first_image_file).unsqueeze(0).to(graphics_card_device)
        second_image_tensor = image_transformation_pipeline(second_image_file).unsqueeze(0).to(graphics_card_device)
        
        # Disables gradient tracking to reduce memory usage.
        with torch.no_grad():
            
            # Passes the images through Zhongyu's network.
            # We do not need to strip a layer here because his model natively outputs embeddings.
            first_face_embedding = face_verification_model(first_image_tensor)
            second_face_embedding = face_verification_model(second_image_tensor)
            
            # Calculates the Euclidean distance and converts it into a similarity score.
            euclidean_distance = torch.nn.functional.pairwise_distance(first_face_embedding, second_face_embedding)
            euclidean_similarity_score = 1.0 / (1.0 + euclidean_distance)
            extracted_euclidean_number = euclidean_similarity_score.item()
            
            # Calculates the Cosine similarity.
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
    pyplot.title("System Evaluation: Zhongyu ResNet34 (Metric Learning)")
    pyplot.legend()

    # Physically saves the final graph as an image file directly to the reports folder.
    final_image_filename = "reports/Zhongyu_ResNet34_ROC_Curve.png"
    pyplot.savefig(final_image_filename, bbox_inches='tight')
    
    print(f"\nEvaluation complete. Graph successfully saved to: {final_image_filename}")