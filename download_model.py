import os
from sentence_transformers import SentenceTransformer

def download_model():
    model_name = "all-MiniLM-L6-v2"
    # Save to the 'models' directory which is mapped in docker-compose
    output_path = os.path.join("models", model_name)
    
    print(f"Downloading {model_name} to {output_path}...")
    
    # This will download the model and save it locally
    model = SentenceTransformer(model_name)
    model.save(output_path)
    
    print(f"Model saved successfully to {output_path}")

if __name__ == "__main__":
    download_model()
