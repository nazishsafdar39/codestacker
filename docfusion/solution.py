import os
import json
import gc
from src.pipeline import DocFusionPipeline

# Batch size for GPU inference — tune based on available memory.
# 4 is a safer default to pass strict autograder memory constraints.
BATCH_SIZE = 4


class DocFusionSolution:
    def __init__(self):
        self.pipeline = DocFusionPipeline()

    def train(self, train_dir: str, work_dir: str) -> str:
        """
        Train a model on data in train_dir.
        """
        model_save_path = os.path.join(work_dir, "saved_models")
        os.makedirs(model_save_path, exist_ok=True)

        self.pipeline.train(train_dir, model_save_path)

        return model_save_path

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        """
        Run inference and write predictions to out_path.
        Uses batched Donut inference for higher throughput.
        """
        self.pipeline.load(model_dir)

        test_info_path = os.path.join(data_dir, "test.jsonl")
        images_dir = os.path.join(data_dir, "images")

        if not os.path.exists(test_info_path):
            print(f"Test info file not found at {test_info_path}, skipping.")
            return

        # ---- Read all records first ----
        records = []
        with open(test_info_path, "r") as f:
            for line in f:
                record = json.loads(line)
                doc_id = record["id"]
                image_path = self._resolve_image(images_dir, doc_id)
                records.append({"id": doc_id, "image_path": image_path})

        # ---- Batched extraction ----
        with open(out_path, "w") as out_f:
            for batch_start in range(0, len(records), BATCH_SIZE):
                batch = records[batch_start : batch_start + BATCH_SIZE]
                batch_paths = [r["image_path"] for r in batch]

                # Run batch extraction through Donut
                extractions = self.pipeline.extract_batch(batch_paths)

                # Run anomaly detection per document (lightweight, no batching needed)
                for rec, (vendor, date, total) in zip(batch, extractions):
                    is_forged = self.pipeline.predict_anomaly(
                        vendor, date, total, rec["image_path"]
                    )

                    result = {
                        "id": rec["id"],
                        "vendor": vendor,
                        "date": date,
                        "total": str(total) if total else None,
                        "is_forged": is_forged,
                    }
                    out_f.write(json.dumps(result) + "\n")

            # Free memory after all batches are processed
            self.pipeline.clear_cache()
            gc.collect()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_image(images_dir: str, doc_id: str) -> str:
        """Resolve image path, checking common extensions."""
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = os.path.join(images_dir, f"{doc_id}{ext}")
            if os.path.exists(candidate):
                return candidate
        return os.path.join(images_dir, f"{doc_id}.jpg")
