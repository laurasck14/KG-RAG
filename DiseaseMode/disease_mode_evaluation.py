import os, json, argparse
import pandas as pd

from llama_index.core import Settings
from llama_index.core.vector_stores.types import MetadataFilters, FilterOperator
from llama_index.core.vector_stores.simple import VectorStoreQuery
from pyhpo import Ontology
from typing import List
from tqdm import tqdm
from src.PrimeKG import PrimeKG

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DiseaseModeEvaluator(PrimeKG):
    """
    Disease Mode Evaluation class that inherits from PrimeKG.
    Provides functionality to evaluate RAG vs No-RAG results against a database.
    """
    
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.dataset_name = None
        self.rag_results = None
        self.no_rag_results = None

    def load_data(self, dataset_file, rag_results_file, no_rag_results_file):
        """Load the dataset and the RAG and No-RAG results."""
        print("Loading evaluation data...")

        self.dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        
        with open(dataset_file, 'r') as f:
            self.dataset = json.load(f)

        with open(rag_results_file, 'r') as f:
            self.rag_results = json.load(f)

        with open(no_rag_results_file, 'r') as f:
            self.no_rag_results = json.load(f)

        print(f"✓ Loaded {len(self.dataset)} dataset entries | name: {self.dataset_name}")
        print(f"✓ Loaded {len(self.rag_results)} RAG results")
        print(f"✓ Loaded {len(self.no_rag_results)} No-RAG results")

    def find_HPO_embedding(self, symptoms: List[str]) -> List[str]:
        """Map LLM symptom responses to HPO terms.  to embeddings using vector and graph stores."""
        if not self.vector_store or not self.graph_store:
            print("Error: Vector store or graph store not initialized")
            return symptoms

        phenotype_dict = {
            "key": "node_type",
            "value": "effect/phenotype",
            "operator": FilterOperator.EQ
        }
        phenotype_filter = MetadataFilters(filters=[phenotype_dict])
        new_symptoms = []
        
        for term in symptoms:
            term = term.strip()
            try: 
                if Ontology.get_hpo_object(term.capitalize()): # if HPOterm exist, use that one
                    new_symptoms.append(term.capitalize())
            except RuntimeError: # if no HPOterm, embed it and return the HPOterm with the highest cosine sim using PrimeKG
                query_embedding = Settings.embed_model.get_text_embedding(term)
                vector_results = self.vector_store.query(
                    VectorStoreQuery(
                        query_embedding=query_embedding, 
                        similarity_top_k=1,
                        filters=phenotype_filter,
                    )
                )
                if not vector_results.ids or len(vector_results.ids) == 0:
                    print(f"Warning: No vector IDs found for term '{term}'")
                    continue
                
                kg_node = self.graph_store.get(ids=[vector_results.ids[0]])
                if kg_node and len(kg_node) > 0:
                    hpo_name = kg_node[0].properties['node_name'] if kg_node else None
                    if vector_results.similarities[0] > 0.5 and kg_node: ## REVISE, SOME KIND OF THRESHOLD?
                        new_symptoms.append(hpo_name)
                else:
                    print(f"Term: {term} | No KG node found for vector ID: {vector_results.ids[0]}")
                    continue
        return list(set(new_symptoms))

    def evaluate_disease_mode(self):
        """
        Evaluate the Disease Mode results by comparing RAG and No-RAG results against a dataset.

        Returns:
            tuple: (results_dataframe, diseases_with_no_results)
        """
        if not all([self.dataset, self.rag_results, self.no_rag_results]):
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if not self.vector_store or not self.graph_store:
            raise ValueError("Stores not initialized. Call start_services() and get stores first.")

        results = []
        no_results = []
        _ = Ontology()  # Initialize the HPO ontology

        print("Running disease mode evaluation...")
        for disease in tqdm(list(self.dataset.keys()), desc="Processing diseases"):
            rag_result = self.rag_results.get(disease, None) 
            no_rag_result = self.no_rag_results.get(disease, None) 

            if (rag_result is not None and no_rag_result is not None and rag_result.get('symptoms') and no_rag_result.get('symptoms')):
                # Ensure HPO terms are used for symptoms
                rag_result['symptoms'] = self.find_HPO_embedding(rag_result['symptoms'])
                no_rag_result['symptoms'] = self.find_HPO_embedding(no_rag_result['symptoms'])

                common_symptoms = set(r.lower() for r in rag_result['symptoms']).intersection(
                    set(n.lower() for n in no_rag_result['symptoms'])) if rag_result and no_rag_result else []
                
                rag_matches = set(s.lower() for s in rag_result['symptoms']).intersection(
                    set(d.lower() for d in self.dataset[disease])) if rag_result else set()

                no_rag_matches = set(s.lower() for s in no_rag_result['symptoms']).intersection(
                    set(d.lower() for d in self.dataset[disease])) if no_rag_result else set()

                # Count parent matches for RAG
                count_rag = set()
                for symptom in rag_result['symptoms']:
                    try: 
                        hpo_term = Ontology.get_hpo_object(symptom) #ensure all terms are valid HPO
                    except RuntimeError:
                        hpo_term = None
                        continue
                    if hpo_term and hpo_term.parents:
                        for parent in hpo_term.parents:
                            parent_name = parent.name
                            if parent_name.lower() in [d.lower() for d in self.dataset[disease]] and parent_name.lower() not in count_rag: # check for matches in response
                                count_rag.add(parent_name.lower())

                # Count parent matches for No-RAG
                count_no_rag = set()
                for symptom in no_rag_result['symptoms']:
                    try: 
                        hpo_term = Ontology.get_hpo_object(symptom)
                    except RuntimeError:
                        hpo_term = None
                    if hpo_term and hpo_term.parents:
                        for parent in hpo_term.parents:
                            parent_name = parent.name
                            if parent_name.lower() in [d.lower() for d in self.dataset[disease]] and parent_name.lower() not in count_no_rag: #check for matches in response
                                count_no_rag.add(parent_name.lower())

                rag_matches = len(rag_matches) + len(count_rag) # add the matches from the parents
                no_rag_matches = len(no_rag_matches) + len(count_no_rag)

                # Calculate precision and recall
                rag_precision = rag_matches / len(rag_result['symptoms']) if len(rag_result['symptoms']) > 0 else 0.0
                rag_recall = rag_matches / len(self.dataset[disease]) if len(self.dataset[disease]) > 0 else 0.0
                no_rag_precision = no_rag_matches / len(no_rag_result['symptoms']) if len(no_rag_result['symptoms']) > 0 else 0.0
                no_rag_recall = no_rag_matches / len(self.dataset[disease]) if len(self.dataset[disease]) > 0 else 0.0

                results.append({
                    "disease": disease.strip().replace("\"", ""),
                    "rag_top_node_id": rag_result['top_node_id'] if rag_result['top_node_id'] else None, # top node ID from PrimeKG
                    "common_symptoms": len(common_symptoms), # matches between the RAG and no-RAG
                    "total_symptoms": len(self.dataset[disease]), # total symptoms in the dataset for a disease
                    "rag_total_symptoms": len(rag_result['symptoms']), # total symptoms in the RAG response
                    "rag_matches": rag_matches, # matches between the RAG response and the dataset
                    "rag_hallucinations": len(rag_result['symptoms']) - rag_matches, # number of not matched symptoms in the RAG
                    "no_rag_total_symptoms": len(no_rag_result['symptoms']), # total symptoms in the No-RAG response
                    "no_rag_matches": no_rag_matches, # matches between the No-RAG response and the dataset
                    "no_rag_hallucinations": len(no_rag_result['symptoms']) - no_rag_matches, # number of not matched symptoms in the No-RAG
                    "rag_accuracy": rag_matches / len(self.dataset[disease]) if len(self.dataset[disease]) > 0 else 0.0,
                    "no_rag_accuracy": no_rag_matches / len(self.dataset[disease]) if len(self.dataset[disease]) > 0 else 0.0,
                    "rag_precision": rag_precision,
                    "rag_recall": rag_recall,
                    "no_rag_precision": no_rag_precision,
                    "no_rag_recall": no_rag_recall,
                })
            else:
                no_results.append(disease)
                continue
        
        return pd.DataFrame(results), no_results

    def run_evaluation(self, dataset_file, rag_results_file, no_rag_results_file, outdir=None):
        """
        Complete evaluation pipeline.
        
        Args:
            dataset_file (str): Path to dataset file
            rag_results_file (str): Path to RAG results
            no_rag_results_file (str): Path to No-RAG results
            outdir (str): Output directory for results
        """
        if outdir is None:
            outdir = os.path.expanduser('~/scratch-llm/results/disease_mode/')

        os.makedirs(outdir, exist_ok=True)

        try:
            # Load data
            self.load_data(dataset_file, rag_results_file, no_rag_results_file)
            
            # Start services and get stores
            if not self.start_services():
                raise Exception("Failed to start NebulaGraph services")
            
            # Get connection pool for session management
            connection_pool = self.get_connection_pool()
            if not connection_pool:
                raise Exception("Failed to get connection pool")
            
            # Create session and get stores
            with connection_pool.session_context('root', 'nebula') as session:
                # Test connection
                result = session.execute('SHOW SPACES;')
                print(f"Connected to NebulaGraph: {result}")
                
                # Get graph and vector stores
                graph_store = self.get_graph_store()
                vector_store = self.get_vector_store()
                
                if not graph_store or not vector_store:
                    raise Exception("Failed to initialize stores")
                
                print("✓ All stores initialized successfully")
                
                # Run evaluation
                results_df, no_results = self.evaluate_disease_mode()
                
                # Save results
                results_file = os.path.join(outdir, f'{self.dataset_name}_results.csv')
                no_results_file = os.path.join(outdir, f'{self.dataset_name}_no_results.txt') # the evaluated diseases is either not in RAG or no-RAG results

                results_df.to_csv(results_file, index=False)
                print(f"✓ Results saved to: {results_file}")
                
                with open(no_results_file, 'w') as f:
                    for disease in no_results:
                        clean_disease = disease.strip().replace('"', '')
                        f.write(f"{clean_disease}\n")
                print(f"✓ No results list saved to: {no_results_file}")
                
                print(f"✓ Evaluation complete! Processed {len(results_df)} diseases successfully")
                print(f"  {len(no_results)} diseases had no results in RAG/No-RAG data")
                
                return results_df, no_results
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise
        finally:
            # Always cleanup
            self.stop_services()


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiseaseMode results using PrimeKG")
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to the data file to run the evaluation with, data/phenopackets/phenopackets_diseases.json or data/Orphanet/orphanet*.json')
    parser.add_argument('--rag-results', type=str, required=True, 
                       help='Path to the RAG results file')
    parser.add_argument('--no-rag-results', type=str, required=True, 
                       help='Path to the No-RAG results file')
    parser.add_argument('--outdir', type=str, 
                       default=os.path.expanduser('~/scratch-llm/results/disease_mode/evaluation'),
                       help='Output directory for results')
    
    args = parser.parse_args()

    # Validate input files
    for file_path in [args.data, args.rag_results, args.no_rag_results]:
        if not os.path.exists(file_path):
            print(f"Error: The file {file_path} does not exist.")
            exit(1)

    # Run evaluation
    evaluator = DiseaseModeEvaluator()
    try:
        results_df, no_results = evaluator.run_evaluation(
            args.data, 
            args.rag_results, 
            args.no_rag_results,
            args.outdir
        )
        print("✓ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()