import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# This defines the exact blueprint of the network required to load the saved weights.
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


# This class handles all the complex PyTorch logic so the frontend user interface does not have to.
class PatrickFaceVerifier:
    def __init__(self, trained_model_weights_path="models/verification_classification_patrick.pt"):
        
        # This specifies the exact number of identities used during the training phase.
        total_training_identities = 4000

        # This builds the empty network architecture in memory.
        self.face_verification_model = MobileNetClassificationNetwork(total_unique_identities=total_training_identities)

        # This loads the physical weights into the architecture.
        model_weights = torch.load(trained_model_weights_path)
        self.face_verification_model.load_state_dict(model_weights)

        # This strips off the final guessing layer to expose the 1024-number embedding.
        self.face_verification_model.base_feature_extractor.classifier[3] = nn.Identity()

        # This sets the network to evaluation mode.
        self.face_verification_model.eval()

        # This transfers the model to the graphics card if one is available.
        self.graphics_card_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_verification_model.to(self.graphics_card_device)

        # This defines the mathematical image transformations required by the model.
        self.image_transformation_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate_face_embedding(self, numpy_face_image_array):
        # This converts the raw webcam array into a mathematical tensor.
        pillow_image_format = Image.fromarray(numpy_face_image_array).convert('RGB')
        transformed_image_tensor = self.image_transformation_pipeline(pillow_image_format).unsqueeze(0).to(self.graphics_card_device)
        
        # This passes the image through the network to generate the 1024 numbers.
        with torch.no_grad():
            extracted_face_embedding = self.face_verification_model(transformed_image_tensor)
            
        return extracted_face_embedding

    def compare_embeddings(self, live_webcam_embedding, saved_database_embedding, approval_threshold=0.65):
        # This calculates the Cosine similarity angle between the two sets of 1024 numbers.
        cosine_similarity_score = torch.nn.functional.cosine_similarity(live_webcam_embedding, saved_database_embedding).item()
        
        # This determines if the score is high enough to unlock the door.
        is_approved_match = cosine_similarity_score >= approval_threshold
        
        # This returns the final decision to the user interface.
        return {
            "match": is_approved_match,
            "similarity_score": round(cosine_similarity_score, 4),
            "distance_metric": "cosine",
            "method": "supervised_classification"
        }