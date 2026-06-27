import sys
import tempfile
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.event_engine import EventEngine
from core.vector_db import VectorDBManager

def run_benchmark():
    print("==================================================")
    print("       VisionGuard Vector DB Benchmark Run        ")
    print("==================================================")

    dim = 128
    num_vectors = 1000
    num_queries = 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "benchmark.db"
        index_path = Path(tmpdir) / "benchmark.faiss"

        # Initialize
        event_engine = EventEngine(db_path)
        vector_db = VectorDBManager(index_path, dim, event_engine)

        # Generate synthetic data
        print(f"[PREPARE] Generating {num_vectors} synthetic face embeddings...")
        embeddings = np.random.rand(num_vectors, dim).astype('float32')
        # Normalize vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Benchmark Insertion
        print(f"[INSERT] Inserting {num_vectors} vectors into hybrid DB...")
        start_insert = time.perf_counter()
        
        for idx in range(num_vectors):
            person_id = f"PERS_{idx:04d}"
            # Register person
            event_engine.register_person(
                person_id=person_id,
                name=f"Mock Person {idx}",
                category="Watchlist",
                risk_level="High"
            )
            # Add embedding
            vector_db.add_face(
                person_id=person_id,
                embedding=embeddings[idx],
                image_path=f"mock_path_{idx}.jpg",
                source_type="manual_enrollment",
                metadata={"notes": f"Benchmark crop {idx}"}
            )
            
        end_insert = time.perf_counter()
        total_insert_time = end_insert - start_insert
        avg_insert_time_ms = (total_insert_time / num_vectors) * 1000.0
        
        print(f"[SUCCESS] Insertion complete.")
        print(f"  - Total Insertion Time: {total_insert_time:.4f} seconds")
        print(f"  - Average Latency per Vector: {avg_insert_time_ms:.4f} ms")

        # Save to disk benchmark
        start_save = time.perf_counter()
        vector_db.save()
        end_save = time.perf_counter()
        save_time_ms = (end_save - start_save) * 1000.0
        file_size_kb = index_path.stat().st_size / 1024.0

        print(f"[DISK] Persisted index saved.")
        print(f"  - Save Latency: {save_time_ms:.2f} ms")
        print(f"  - FAISS File Size: {file_size_kb:.2f} KB ({index_path.stat().st_size} bytes)")

        # Benchmark Querying
        print(f"[QUERY] Executing {num_queries} random vector queries...")
        query_vectors = np.random.rand(num_queries, dim).astype('float32')
        query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_vectors = query_vectors / query_norms

        start_query = time.perf_counter()
        
        for idx in range(num_queries):
            # Query Top-5 closest vectors with a threshold of 0.30
            vector_db.search(
                embedding=query_vectors[idx],
                k=5,
                threshold=0.30,
                query_source="benchmark_query"
            )
            
        end_query = time.perf_counter()
        total_query_time = end_query - start_query
        avg_query_time_ms = (total_query_time / num_queries) * 1000.0
        qps = num_queries / total_query_time

        print(f"[SUCCESS] Query benchmark complete.")
        print(f"  - Total Search Time: {total_query_time:.4f} seconds")
        print(f"  - Average Search Latency: {avg_query_time_ms:.4f} ms")
        print(f"  - Throughput: {qps:.2f} QPS (Queries Per Second)")
        print("==================================================")

if __name__ == "__main__":
    run_benchmark()
