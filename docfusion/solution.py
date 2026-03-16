import os
import json
import joblib
from src.pipeline import DocFusionPipeline

class DocFusionSolution:
    def __init__(self):
        self.pipeline = DocFusionPipeline()

    def train(self, train_dir: str, work_dir: str) -> str:
        """
        Train a model on data in train_dir.
        """
        model_save_path = os.path.join(work_dir, "saved_models")
        os.makedirs(model_save_path, exist_ok=True)
        
        # We can implement a method that reads `train.jsonl` from train_dir
        # and fits the anomaly detector or extractor
        self.pipeline.train(train_dir, model_save_path)
        
        return model_save_path

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        """
        Run inference and write predictions to out_path.
        """
        self.pipeline.load(model_dir)
        
        test_info_path = os.path.join(data_dir, "test.jsonl")
        images_dir = os.path.join(data_dir, "images")
        
        # Ensure we can run without actual data for basic validation
        if not os.path.exists(test_info_path):
            print(f"Test info file not found at {test_info_path}, writing dummy predictions if needed.")
            return

        with open(test_info_path, 'r') as f, open(out_path, 'w') as out_f:
            for line in f:
                record = json.loads(line)
                doc_id = record["id"]
                image_path = os.path.join(images_dir, f"{doc_id}.jpg")
                
                # Check if image exists, though in prod it should
                if not os.path.exists(image_path):
                    # Fallback to .png or similar
                    alt_path = os.path.join(images_dir, f"{doc_id}.png")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                
                vendor, date, total = self.pipeline.extract(image_path)
                
                # We need the OCR text for anomaly detection as well 
                # (our heuristic pipeline gives it back but pipeline might hide it)
                # Let's say pipeline.predict_anomaly takes vendor, date, total, image_path
                is_forged = self.pipeline.predict_anomaly(vendor, date, total, image_path)
                
                result = {
                    "id": doc_id,
                    "vendor": vendor,
                    "date": date,
                    "total": str(total) if total else None,
                    "is_forged": is_forged
                }
                out_f.write(json.dumps(result) + "\n")
