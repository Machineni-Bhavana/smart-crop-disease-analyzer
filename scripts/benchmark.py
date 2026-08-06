import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
import numpy as np

from config import Config
from src.inference import DiseasePredictor
from src.utils.logger import get_logger

logger = get_logger("Benchmark")

def run_benchmark(checkpoint_path: Path, test_dir: Path, output_path: Path, max_samples: int = 100) -> dict:
    """
    Run latency benchmark over test images and compute statistical metrics.
    """
    logger.info(f"Loading DiseasePredictor from '{checkpoint_path}' for latency benchmark...")
    predictor = DiseasePredictor(checkpoint_path=checkpoint_path)
    
    # Collect test images
    image_paths = []
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for f in test_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in supported_exts:
            image_paths.append(f)
            
    if not image_paths:
        logger.error(f"No test images found in '{test_dir}' for benchmarking.")
        return {}
        
    image_paths = image_paths[:max_samples]
    logger.info(f"Benchmarking inference latency over {len(image_paths)} test images...")
    
    latencies = []
    
    # Warmup prediction
    _ = predictor.predict(image_paths[0])
    
    for img_path in image_paths:
        res = predictor.predict(img_path)
        latencies.append(res["inference_time_ms"])
        
    latencies_arr = np.array(latencies)
    
    mean_lat = float(np.mean(latencies_arr))
    median_lat = float(np.median(latencies_arr))
    p95_lat = float(np.percentile(latencies_arr, 95))
    min_lat = float(np.min(latencies_arr))
    max_lat = float(np.max(latencies_arr))
    
    sub_2s = bool(p95_lat < 2000.0)
    
    benchmark_report = {
        "model_name": predictor.model_name,
        "num_samples_evaluated": len(image_paths),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "mean_latency_sec": round(mean_lat / 1000.0, 4),
        "p95_latency_sec": round(p95_lat / 1000.0, 4),
        "sub_2_second_compliant": sub_2s
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(benchmark_report, f, indent=4)
        
    logger.info("=== INFERENCE LATENCY BENCHMARK REPORT ===")
    logger.info(f"  - Model: {predictor.model_name}")
    logger.info(f"  - Evaluated Samples: {len(image_paths)}")
    logger.info(f"  - Mean Latency: {mean_lat:.2f} ms ({mean_lat/1000.0:.4f} s)")
    logger.info(f"  - Median Latency: {median_lat:.2f} ms")
    logger.info(f"  - 95th Percentile (p95): {p95_lat:.2f} ms ({p95_lat/1000.0:.4f} s)")
    logger.info(f"  - Sub-2-Second SLA Compliant: {'YES' if sub_2s else 'NO'}")
    logger.info(f"Saved benchmark report to '{output_path}'")
    
    return benchmark_report

def main():
    parser = argparse.ArgumentParser(description="Benchmark inference latency across test set.")
    parser.add_argument("--checkpoint_path", type=str, default=str(Config.BEST_MODEL_PATH))
    parser.add_argument("--test_dir", type=str, default=str(Config.TEST_DIR))
    parser.add_argument("--output_path", type=str, default=str(Config.OUTPUT_DIR / "inference_benchmark.json"))
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=Path(args.checkpoint_path),
        test_dir=Path(args.test_dir),
        output_path=Path(args.output_path),
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
