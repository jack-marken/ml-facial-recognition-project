import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# Author: Patrick (100599029)

# This defines the structure of the neural network using a pre-trained base.
class MobileNetClassificationNetwork(nn.Module):
    def __init__(self, total_unique_identities):
        super(MobileNetClassificationNetwork, self).__init__()

        # This layer downloads the complete MobileNetV2 architecture.
        # It loads the standard pre-calculated weights from the ImageNet dataset.
        self.base_feature_extractor = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # This loop iterates through every individual layer in the downloaded base model.
        for network_parameter in self.base_feature_extractor.parameters():
            # This explicitly prevents the base layer weights from being modified during this training phase.
            network_parameter.requires_grad = False

        # This variable accesses the classification section of the base model.
        # It extracts the exact number of mathematical features outputted by the final base layer.
        number_of_input_features = self.base_feature_extractor.classifier[1].in_features

        # This layer creates an intermediate step to process the features before the final classification.
        # It maps the inputs to 1024 intermediate features to find complex correlations.
        self.intermediate_fully_connected_layer = nn.Linear(
            in_features=number_of_input_features, 
            out_features=1024,
            bias=True
        )
        
        # This function applies the Rectified Linear Unit non-linearity to the intermediate layer output.
        self.intermediate_activation_function = nn.ReLU()

        # This layer randomly disables 50 percent of the connections during training.
        # This forces the network to learn robust features and prevents overfitting to the training data.
        self.dropout_regularisation_layer = nn.Dropout(
            p=0.5, 
            inplace=False
        )

        # This final layer takes the 1024 intermediate features and calculates a specific prediction score.
        # It outputs one score for each of the unique identities in the dataset.
        self.final_classification_layer = nn.Linear(
            in_features=1024, 
            out_features=total_unique_identities, 
            bias=True
        )

        # This overrides the default classification head of the MobileNetV2 model.
        # It replaces it with the newly defined sequence of expanded layers.
        self.base_feature_extractor.classifier = nn.Sequential(
            self.intermediate_fully_connected_layer,
            self.intermediate_activation_function,
            self.dropout_regularisation_layer,
            self.final_classification_layer
        )

    def forward(self, input_image_batch):
        # This block passes the input images through the frozen convolutional layers.
        # It then automatically passes the extracted features through the expanded classification head.
        final_prediction_output = self.base_feature_extractor(input_image_batch)
        
        # This line returns the final calculated prediction scores for the batch of images.
        return final_prediction_output

# This block is required by Windows to safely execute multiprocessing with background CPU workers.
if __name__ == '__main__':

    # Defines relative paths to the dataset folders.
    training_directory_path = "datasets/classification_data/train_data"
    validation_directory_path = "datasets/classification_data/val_data"

    # Configures the sequence of image transformations.
    # Resizes images to 224 by 224 pixels, converts them to tensors, and normalises the colour channels.
    image_transformation_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loads training data using the directory structure.
    training_image_dataset = datasets.ImageFolder(
        root=training_directory_path,
        transform=image_transformation_pipeline
    )

    # Loads the validation data using the directory structure.
    validation_image_dataset = datasets.ImageFolder(
        root=validation_directory_path,
        transform=image_transformation_pipeline
    )

    # This determines the number of background processes used to load data from the hard drive.
    cpu_worker_processes = 4

    # This defines the number of images processed simultaneously in a single forward pass to maximise memory utilisation.
    increased_batch_size = 256

    # Creates an iterable loader for the training data to process images in optimised batches.
    # The pin_memory setting pre-allocates memory to accelerate data transfer to the computation device.
    training_data_loader = DataLoader(
        dataset=training_image_dataset,
        batch_size=increased_batch_size,
        shuffle=True,
        num_workers=cpu_worker_processes,
        pin_memory=True
    )

    # Creates an iterable loader for the validation data.
    validation_data_loader = DataLoader(
        dataset=validation_image_dataset,
        batch_size=increased_batch_size,
        shuffle=False,
        num_workers=cpu_worker_processes,
        pin_memory=True
    )

    # Determines the total number of unique identity classes based on the folder names.
    total_unique_identities = len(training_image_dataset.classes)

    # Outputs the dataset metrics to the console.
    print(f"Total training images loaded: {len(training_image_dataset)}")
    print(f"Total validation images loaded: {len(validation_image_dataset)}")
    print(f"Total unique identities to classify: {total_unique_identities}")

    # This creates a new instance of the modified MobileNetV2 network.
    face_recognition_model = MobileNetClassificationNetwork(total_unique_identities=total_unique_identities)

    # This prints the entire architecture of the network to the terminal to verify its construction.
    print(face_recognition_model)

    # This defines the loss calculation method for multi-class classification.
    classification_loss_function = nn.CrossEntropyLoss()

    # This initialises the optimiser to update the network weights during training.
    network_optimizer = optim.Adam(face_recognition_model.parameters(), lr=0.001)

    # This initialises the gradient scaler for Automatic Mixed Precision to increase processing speed.
    gradient_scaler = GradScaler('cuda')

    # This defines the number of full passes through the training dataset.
    total_training_epochs = 10

    # This checks if a Graphics Processing Unit is available on the system.
    if torch.cuda.is_available():
        # This assigns the computation device to the GPU.
        computation_device = torch.device("cuda")
    else:
        # This assigns the computation device to the CPU.
        computation_device = torch.device("cpu")

    # This moves the model to the assigned computation device.
    face_recognition_model.to(computation_device)

    # This prints the selected computation device to the terminal.
    print("Starting training on device:", computation_device)

    # This loops through the dataset for the specified number of epochs.
    for current_epoch_number in range(total_training_epochs):
        
        # This sets the model to training mode to enable the dropout layer.
        face_recognition_model.train()
        
        # This initialises a variable to track the accumulated loss for the current epoch.
        running_epoch_loss_total = 0.0

        # This creates a visual progress bar for the training data loader.
        epoch_progress_bar = tqdm(training_data_loader, desc=f"Epoch {current_epoch_number + 1}/{total_training_epochs}")

        # This iterates over the batches of images and labels provided by the progress bar wrapper.
        for batch_index, (image_batch_tensor, label_batch_tensor) in enumerate(epoch_progress_bar):
            
            # This moves the input images to the designated computation device.
            image_batch_tensor = image_batch_tensor.to(computation_device)
            
            # This moves the input labels to the designated computation device.
            label_batch_tensor = label_batch_tensor.to(computation_device)

            # This clears the old gradients from the previous optimisation step.
            network_optimizer.zero_grad()

            # This enables Automatic Mixed Precision to compute in 16-bit.
            with autocast('cuda'):
                # This passes the batch of images through the model to generate predictions.
                prediction_outputs = face_recognition_model(image_batch_tensor)

                # This calculates the error between the model predictions and the actual labels.
                batch_error_loss = classification_loss_function(prediction_outputs, label_batch_tensor)

            # This scales the loss and computes the gradients.
            gradient_scaler.scale(batch_error_loss).backward()

            # This un-scales the gradients and updates the model parameters.
            gradient_scaler.step(network_optimizer)

            # This updates the scale factor for the next iteration.
            gradient_scaler.update()

            # This adds the current batch loss to the running total.
            running_epoch_loss_total = running_epoch_loss_total + batch_error_loss.item()
            
            # This updates the visual progress bar with the current batch error loss.
            epoch_progress_bar.set_postfix(loss=batch_error_loss.item())

        # This calculates the average loss across the entire epoch.
        average_epoch_loss = running_epoch_loss_total / len(training_data_loader)
        
        # This prints the final average loss for the completed epoch.
        print(f"--- Epoch {current_epoch_number + 1} completed. Average Loss: {average_epoch_loss:.4f} ---")

    # This saves the trained model weights to a file for later use in verification.
    torch.save(face_recognition_model.state_dict(), "models/verification_classification_patrick.pt")

    # This prints a final confirmation message.
    print("Training complete. Model saved to models/verification_classification_patrick.pt")