from sentence_transformers import SentenceTransformer
import yaml

def get_embedding_model(config_path="config.yaml"):
    """
    Loads the sentence-transformer model specified in the config.
    """
    print("--- Loading Embedding Model ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config.get('rag', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        print(f"Loading model: {model_name}")
        
        # This will download the model from Hugging Face the first time
        model = SentenceTransformer(model_name)
        
        print("--- Embedding Model Loaded Successfully ---")
        return model
        
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

if __name__ == "__main__":
    # Test if the model loads correctly
    model = get_embedding_model()
    if model:
        print("\nModel loaded. Testing embedding:")
        test_sentence = "This is a test for Cortexa."
        embedding = model.encode(test_sentence)
        print(f"Test sentence: '{test_sentence}'")
        print(f"Embedding dimension: {len(embedding)}")
        print("Test successful!")