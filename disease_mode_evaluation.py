import os, json, pickle, time, argparse
import pandas as pd

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStoreData, SimpleVectorStore, VectorStoreQuery
from llama_index.core.vector_stores.types import MetadataFilters, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from pyhpo import Ontology
from typing import List
from nebulagraph_lite import nebulagraph_let as ng_let
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_nebulagraph_containers():
    """
    Ensure that the necessary NebulaGraph containers are set up and started.
    This function uses udocker to pull, create, and set up the NebulaGraph containers.
    """

    # Ensure all necessary containers are created, set up, and started
    services = ['nebula-metad', 'nebula-graphd', 'nebula-storaged']
    for service in services:
        os.system(f'udocker pull vesoft/{service}:v3')
        os.system(f'udocker create --name={service} vesoft/{service}:v3')
        os.system(f'udocker setup --execmode=F1 {service}')

def load_data(phenopackets_file, rag_results_file, no_rag_results_file):
    """
    Load the phenopackets data and the RAG and No-RAG results.
    """
    with open(phenopackets_file, 'r') as f:
        phenopackets = json.load(f)

    with open(rag_results_file, 'r') as f:
        rag_results = json.load(f)

    with open(no_rag_results_file, 'r') as f:
        no_rag_results = json.load(f)

    return phenopackets, rag_results, no_rag_results

def setup_vector_store():
    """
    Set up the vector store with pre-embedded nodes.
    """
    # Load the pre-embedded nodes from a file
    with open(os.path.expanduser('~/scratch-llm/storage/nodes/all_nodes_all-mpnet-base-v2.pkl'), 'rb') as f:
        all_nodes_embedded: List[TextNode] = pickle.load(f)

    # Initialize the embedding model and create dictionaries from the nodes
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

    embedding_dict = {node.id_: node.get_embedding() for node in all_nodes_embedded}
    text_id_to_ref_doc_id = {node.id_: node.ref_doc_id or "None" for node in all_nodes_embedded}
    metadata_dict = {node.id_: node.metadata for node in all_nodes_embedded}

    # Initialize the SimpleVectorStore with the dictionaries
    vector_store = SimpleVectorStore(
        data = SimpleVectorStoreData(
            embedding_dict=embedding_dict,
            text_id_to_ref_doc_id=text_id_to_ref_doc_id,
            metadata_dict=metadata_dict,
        ),
        stores_text=True
    )

## MAP HPO terms to embeddings
def find_HPO_embedding(symptoms: List[str], vector_store, graph_store) -> List[str]:
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
            if Ontology.get_hpo_object(term.capitalize()):
                new_symptoms.append(term.capitalize())

        except RuntimeError:
            query_embedding = Settings.embed_model.get_text_embedding(term)
            vector_results = vector_store.query(
                VectorStoreQuery(
                    query_embedding=query_embedding, 
                    similarity_top_k=1,
                    filters=phenotype_filter,
                )
            )
            if not vector_results.ids or len(vector_results.ids) == 0:
                print(f"Warning: No vector IDs found for term '{term}'")
                continue
            
            kg_node = graph_store.get(ids=[vector_results.ids[0]])
            if kg_node and len(kg_node) > 0:
                hpo_name = kg_node[0].properties['node_name'] if kg_node else None
                # print(f"Term: {term} | Vector ID: {vector_results.ids[0]} | KG node: {hpo_name} | Similarity: {vector_results.similarities[0]:.4f}")
                if vector_results.similarities[0] > 0.5 and kg_node:
                    new_symptoms.append(hpo_name)
            else:
                print(f"Term: {term} | No KG node found for vector ID: {vector_results.ids[0]}")
                continue
    return list(set(new_symptoms))

def DiseaseModeEvaluation(phenopackets, rag_results, no_rag_results, vector_store, graph_store):
    """
    Evaluate the Disease Mode results by comparing RAG and No-RAG results against a database (phenopacket data).
    """
    results = []
    no_results = []
    _ = Ontology()  # Initialize the HPO ontology

    for disease in tqdm(list(phenopackets.keys()), desc="Processing diseases"):
        # find the disease in the rag and no_rag results

        rag_result = rag_results.get(disease, None) 
        no_rag_result = no_rag_results.get(disease, None) 
        
        if rag_result is not None and no_rag_result is not None:
            # ensure HPO terms are used for symptoms
            rag_result['symptoms'] = find_HPO_embedding(rag_result['symptoms'], vector_store, graph_store) if rag_result['symptoms'] else []
            no_rag_result['symptoms'] = find_HPO_embedding(no_rag_result['symptoms'], vector_store, graph_store) if no_rag_result['symptoms'] else []

            common_symptoms = set(r.lower() for r in rag_result['symptoms']).intersection(set(n.lower() for n in no_rag_result['symptoms'])) if rag_result and no_rag_result else []
            rag_matches = set(s.lower() for s in rag_result['symptoms']).intersection(set(d.lower() for d in phenopackets[disease])) if rag_result else set()
            no_rag_matches = set(s.lower() for s in no_rag_result['symptoms']).intersection(set(d.lower() for d in phenopackets[disease])) if no_rag_result else set()

            
            count_rag = 0
            for symptom in rag_result['symptoms']:
                try: 
                    hpo_term = Ontology.get_hpo_object(symptom)
                except RuntimeError:
                    hpo_term = None
                if hpo_term:
                    if hpo_term.parents:
                        for parent in hpo_term.parents:
                            parent_name = parent.name

                            # check if parent_name is in the phenopackets data
                            if parent_name.lower() in [d.lower() for d in phenopackets[disease]]:
                                count_rag += 1
                    else:
                        continue
                else:
                    continue
            
            count_no_rag = 0
            for symptom in no_rag_result['symptoms']:
                try: 
                    hpo_term = Ontology.get_hpo_object(symptom)
                except RuntimeError:
                    hpo_term = None
                if hpo_term:
                    if hpo_term.parents:
                        for parent in hpo_term.parents:
                            parent_name = parent.name

                            # check if parent_name is in the phenopackets data
                            if parent_name.lower() in [d.lower() for d in phenopackets[disease]]:
                                count_no_rag += 1
                    else:
                        continue
                else:
                    continue

            #calculate precision and recall
            rag_precision = len(rag_matches) / len(rag_result['symptoms']) if len(rag_result['symptoms']) > 0 else 0.0
            rag_recall = len(rag_matches) / len(phenopackets[disease]) if len(phenopackets[disease]) > 0 else 0.0
            no_rag_precision = len(no_rag_matches) / len(no_rag_result['symptoms']) if len(no_rag_result['symptoms']) > 0 else 0.0
            no_rag_recall = len(no_rag_matches) / len(phenopackets[disease]) if len(phenopackets[disease]) > 0 else 0.0

            # calcuate the average symptom coverage
            symptom_coverage = len(set(phenopackets[disease]) - set(rag_result['symptoms']) - set(no_rag_result['symptoms'])) / len(set(phenopackets[disease])) if len(set(phenopackets[disease])) > 0 else 0.0
            results.append({
                "disease": disease.strip().replace("\"", ""),
                "common_symptoms": len(common_symptoms), # matches between rag and no_rag using HPO terms
                "total_symptoms": len(phenopackets[disease]), # total number of symptoms in the evaluation dataset for a disease
                "rag_total_symptoms": len(rag_result['symptoms']), # total number of symptoms in RAG result for a disease
                "rag_matches": len(rag_matches) + count_rag, # matches between rag response and dataset using HPO terms
                "rag_hallucinations": len(rag_result['symptoms']) - len(rag_matches) - count_rag, # hallucinations in RAG results
                "no_rag_total_symptoms": len(no_rag_result['symptoms']), # total number of symptoms in No-RAG result for a disease
                "no_rag_matches": len(no_rag_matches) + count_no_rag, # matches between no_rag response and evaluation dataset using HPO terms
                "no_rag_hallucinations": len(no_rag_result['symptoms']) - len(no_rag_matches) - count_no_rag, # hallucinations in No-RAG results
                "rag_accuracy": len(rag_matches) / len(phenopackets[disease]) if len(phenopackets[disease]) > 0 else 0.0,
                "no_rag_accuracy": len(no_rag_matches) / len(phenopackets[disease]) if len(phenopackets[disease]) > 0 else 0.0,
                "rag_precision": rag_precision,
                "rag_recall": rag_recall,
                "no_rag_precision": no_rag_precision,
                "no_rag_recall": no_rag_recall,
                "symptom_coverage": symptom_coverage, # fraction of unique symptoms that are predicted by neither RAG nor No-RAG
            })
        else:
            # print(f"{disease} was not found in rag or no_rag results")
            no_results.append(disease)
            continue
    return pd.DataFrame(results), no_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Disease Mode results")
    parser.add_argument('--phenopackets', type=str, required=True, help='Path to the phenopackets data file')
    parser.add_argument('--rag_results', type=str, required=True, help='Path to the RAG results file')
    parser.add_argument('--no_rag_results', type=str, required=True, help='Path to the No-RAG results file')
    args = parser.parse_args()

    # validate input arguments
    for file_path in [args.phenopackets, args.rag_results, args.no_rag_results]:
        if not os.path.exists(file_path):
            print(f"Error: The file {file_path} does not exist.")
            exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...", flush=True)
    phenopackets, rag_results, no_rag_results = load_data(args.phenopackets, args.rag_results, args.no_rag_results)

    print("Setting up vector store and embedding model", flush=True)
    vector_store = setup_vector_store()

    print("Setting up NebulaGraph containers", flush=True)



    ## LOADING NEBULA GRAPH 
    os.system('udocker ps')
    n = ng_let(in_container=True)
    n.start() # This takes around 5 mins

    # Configure connection pool
    config = Config()
    config.max_connection_pool_size = 10
    connection_pool = ConnectionPool()

    # Initialize connection pool
    try:
        if not connection_pool.init([('127.0.0.1', 9669)], config):
            raise Exception("Failed to initialize connection pool")
        print("Connection pool initialized successfully")
    except Exception as e:
        print(f"Error initializing connection pool: {e}")
        exit(1)

    # Create a session and execute a query
    try:
        with connection_pool.session_context('root', 'nebula') as session:
            result = session.execute('SHOW SPACES;')
            print(result, flush=True)
            time.sleep(30)

            from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
            print("NebulaPropertyGraphStore", flush=True)
            graph_store = NebulaPropertyGraphStore(
                space= "PrimeKG",
                username = "root",
                password = "nebula",
                url = "nebula://localhost:9669",
                props_schema= """`node_index` STRING, `node_type` STRING, `node_id` STRING, `node_name` STRING, 
                    `node_source` STRING, `mondo_id` STRING, `mondo_name` STRING, `group_id_bert` STRING, 
                    `group_name_bert` STRING, `orphanet_prevalence` STRING, `display_relation` STRING """,
            )

            print("Running DiseaseMode evaluation...", flush=True)
            results, no_results = DiseaseModeEvaluation(phenopackets, rag_results, no_rag_results, vector_store, graph_store)
            results.to_csv(os.path.expanduser('~/scratch-llm/results/disease_mode/phenopackets_results.csv'), index=False)
            with open(os.path.expanduser('~/scratch-llm/results/disease_mode/phenopackets_no_results.txt'), 'w') as f:
                for disease in no_results:
                    f.write(f"{disease.strip().replace('\"', '')}\n")

    finally:
        # Close the connection pool
        n.stop()
        session.release()
        connection_pool.close()
        print("Connection pool closed")

if __name__ == "__main__":
    main()