import os
import sys

def main():
    print("Testing Submission...")
    sub_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    sys.path.insert(0, os.path.abspath(sub_dir))
    
    try:
        from solution import DocFusionSolution
        sol = DocFusionSolution()
        print("✅ DocFusionSolution instantiated")
        
        # Test Train
        os.makedirs("work_dir", exist_ok=True)
        model_dir = sol.train("dummy_data/train", "work_dir")
        print(f"✅ Train completed. Model dir: {model_dir}")
        
        # Test Predict
        sol.predict(model_dir, "dummy_data/test", "predictions.jsonl")
        print("✅ Predict completed.")
        
        # Verify JSONL
        import json
        with open("predictions.jsonl") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                assert "id" in data
                assert "is_forged" in data
                
        print("✅ Output format valid.")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
