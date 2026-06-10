import json
import pandas as pd
import time
import os
import torch
from tqdm import tqdm
from data.loader import DataLoader
from models.retriever import Retriever
from models.baselines import GPT2Baseline, RAGGPT2
from models.ht_rag import HTRAG
from evaluation.metrics import Evaluator

# Configuration
SAMPLE_SIZE = 200 # Sample size for quick demo. Increase to 100+ for real results.
OUTPUT_DIR = "research_study/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_pipeline():
    # 1. Load Data
    loader = DataLoader(sample_size=SAMPLE_SIZE)
    datasets = loader.load_all()
    
    # 2. Initialize Shared Retriever
    # For RAG, we need to index the contexts.
    # Note: Closed-world assumption per dataset.
    retriever_models = {} 
    
    # We need a retriever instance per dataset because the corpus changes
    # OR we rebuild index per dataset loop.
    base_retriever = Retriever()
    
    # 3. Initialize Models
    # We'll instantiate them inside the loop or here.
    # HT-RAG and RAG-GPT2 share the retriever structure.
    
    # 4. Evaluator
    evaluator = Evaluator()
    
    all_results = []
    
    for dataset_name, data in datasets.items():
        print(f"\n=== Processing Dataset: {dataset_name} ===")
        
        # Build Corpus for this dataset
        # For SQuAD/Hotpot, use the contexts in the dataset as the 'Knowledge Base'
        corpus = [d['context'] for d in data if d['context']]
        # Deduplicate
        corpus = list(set(corpus))
        
        print(f"Building Index for {dataset_name} with {len(corpus)} documents...")
        # Re-using the same model, just rebuilding index
        base_retriever.build_index(corpus)
        
        models = {
            'GPT-2': GPT2Baseline(),
            'RAG-GPT2': RAGGPT2(base_retriever),
            'HT-RAG': HTRAG(base_retriever)
        }
        
        for model_name, model in models.items():
            print(f"--- Running Model: {model_name} ---")
            
            predictions = []
            references = []
            contexts = [] # For faithfulness check
            latencies = []
            transparency_scores = []
            
            detailed_logs = []
            
            for item in tqdm(data):
                question = item['question']
                gold_answers = item['answers'] # List of strings
                
                start_time = time.time()
                
                # Run Inference
                if model_name == 'GPT-2':
                    output = model.generate(question)
                    retrieved_ctx = "" # No retrieval
                elif model_name == 'RAG-GPT2':
                    output = model.run(question)
                    retrieved_ctx = output['retrieved_context']
                else: # HT-RAG
                    output = model.run(question)
                    retrieved_ctx = output['retrieved_context']
                
                end_time = time.time()
                latencies.append(end_time - start_time)
                
                pred_text = output['answer']
                predictions.append(pred_text)
                references.append(gold_answers)
                contexts.append(retrieved_ctx)
                
                # Metrics
                if model_name == 'HT-RAG':
                    t_score = evaluator.compute_transparency_score(output)
                else:
                    t_score = 0.0
                transparency_scores.append(t_score)
                
                detailed_logs.append({
                    'id': item['id'],
                    'question': question,
                    'gold_answers': gold_answers,
                    'prediction': pred_text,
                    'context': retrieved_ctx[:200] + "..." if retrieved_ctx else "",
                    'transparency_metadata': output.get('ht_rag_metadata', None)
                })
            
            # Compute Aggregate Metrics
            metrics = evaluator.evaluate_batch(predictions, references, contexts)
            
            # Add system metrics
            metrics['avg_latency'] = sum(latencies) / len(latencies)
            metrics['avg_transparency'] = sum(transparency_scores) / len(transparency_scores)
            
            print(f"Result {model_name} on {dataset_name}: {metrics}")
            
            # Save Summary
            result_entry = {
                'model': model_name,
                'dataset': dataset_name,
                **metrics
            }
            all_results.append(result_entry)
            
            # Save Detailed Logs
            with open(f"{OUTPUT_DIR}/log_{dataset_name}_{model_name}.json", 'w') as f:
                json.dump(detailed_logs, f, indent=2)

    # Save Final CSV
    df = pd.DataFrame(all_results)
    df.to_csv(f"{OUTPUT_DIR}/final_results.csv", index=False)
    print(f"\nexperiment complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_pipeline()
