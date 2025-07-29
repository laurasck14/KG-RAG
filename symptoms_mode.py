import os, pickle, time, json, pathlib, signal, torch
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebulagraph_lite import nebulagraph_let as ng_let
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HOME'] = str(pathlib.Path("~/scratch-llm/storage/cache/huggingface/").expanduser().absolute()) # '/scratch-llm/storage/cache/'
# os.environ["TRANSFORMERS_CACHE"] = "~/scratch-llm/storage/models/"

from transformers import AutoTokenizer
from nebulagraph_lite import nebulagraph_let as ng_let
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.vector_stores.simple import SimpleVectorStoreData, SimpleVectorStore, VectorStoreQuery

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from typing import List
from pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser

print("loading embeddings...")
# Load the actual data into all_nodes_embeddded
with open(os.path.expanduser('~/scratch-llm/storage/nodes/all_nodes_all-mpnet-base-v2.pkl'), 'rb') as f:
    all_nodes_embedded: List[TextNode] = pickle.load(f)
# Create dictionaries from the nodes
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

print("loading tokenizer and llm...")
class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()
signal.signal(signal.SIGALRM, handler)

def safe_llm_call(summarizer, *args, timeout=300, **kwargs):
    signal.alarm(timeout)
    try:
        response = summarizer.get_response(*args, **kwargs)
        signal.alarm(0)
        return response
    except TimeoutException:
        print("LLM call timed out. Skipping this item.")
        return None
    except RecursionError:
        print("RecursionError: LLM summarizer entered an infinite loop. Skipping this item.")
        return None
    finally:
        signal.alarm(0)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", padding_side="left", device_map="auto")    
if tokenizer.pad_token_id is None: #no <pad> token previously defined, only eos_token
    tokenizer.pad_token = "<|end_of_text|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    context_window=8192,
    max_new_tokens=3048,
    generate_kwargs={
        "temperature": 0.10, 
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "top_k": 10, 
        "top_p": 0.9,
        # "repetition_penalty": 0.9,  # Added to reduce repetition
        # "no_repeat_ngram_size": 3,  # Prevents repetition of n-grams
    },
    model_kwargs={
        "torch_dtype": torch.float16,
    },
    tokenizer=tokenizer,
    # device_map="auto",  # Automatically offload layers to CPU if GPU memory is insufficient
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    stopping_ids=[tokenizer.eos_token_id],
    tokenizer_kwargs={"max_length": None},
    is_chat_model=True,
)

Settings.llm = llm
Settings.chunk_size = 1024
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2") # BAAI/bge-small-en-v1.5 /  m3 / sentence-transformers/all-mpnet-base-v2

print("loading symptoms mode...")
from llama_index.core.vector_stores.types import MetadataFilters, FilterOperator
phenotype_dict = {
    "key": "node_type",
    "value": "effect/phenotype",
    "operator": FilterOperator.EQ
}
phenotype_filter = MetadataFilters(filters=[phenotype_dict])

class SymptomsMode():
    def __init__(self, vector_store: SimpleVectorStore, graph_store: NebulaPropertyGraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve(self, query: List[str]):
        if not isinstance(query, list):
            return None
            
        from collections import defaultdict
        
        disease_counter = {}
        total_symptoms = len(query)
        # print(f"Processing {total_symptoms} symptoms: {query}")
        
        # Collect all diseases and count symptom matches
        for symptom in query:
            query_embedding = Settings.embed_model.get_text_embedding(symptom)
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=1,
                filters=phenotype_filter,
            )
            individual_results = vector_store.query(vector_store_query)
            
            for node_id, score in zip(individual_results.ids, individual_results.similarities):
                # Get related diseases from graph
                graph_nodes = graph_store.structured_query(
                    """
                    MATCH (e:Node__) WHERE id(e) == $ids
                    MATCH p=(e)-[r:Relation__{label:"disease-phenotype-positive"}]-(t) 
                    RETURN DISTINCT id(t), t.Props__.node_name, t.Chunk__.text
                    """, 
                    param_map={"ids": node_id}
                )
                
                # Process each related disease
                for node in graph_nodes:
                    disease_id = node['id(t)']
                    disease_name = node['t.Props__.node_name']
                    
                    if disease_id not in disease_counter:
                        disease_counter[disease_id] = {
                            'index': disease_id,
                            'name': disease_name,
                            'count': 1,
                            'symptoms': [symptom]
                        }
                    else:
                        disease_counter[disease_id]['count'] += 1
                        if symptom not in disease_counter[disease_id]['symptoms']:
                            disease_counter[disease_id]['symptoms'].append(symptom)
        
        if not disease_counter:
            print("No diseases found matching any symptoms.")
            return {
                'top_diseases': {},
                'top_match_counts': [],
                'grouped_diseases': {},
                'total_symptoms': total_symptoms,
                'top_diseases_list': []
            }
        
        # Get top 2 match counts
        all_match_counts = sorted(set(data['count'] for data in disease_counter.values()), reverse=True)
        top_match_counts = all_match_counts[:2]
        
        # Filter diseases with top 2 match counts
        top_diseases = {
            disease_id: data for disease_id, data in disease_counter.items()
            if data['count'] in top_match_counts
        }
        
        # Group and sort diseases by match count
        grouped_diseases = defaultdict(list)
        for disease_id, data in top_diseases.items():
            grouped_diseases[data['count']].append((disease_id, data))
        
        for count in grouped_diseases:
            grouped_diseases[count].sort(key=lambda x: x[1]['name'])
        
        # Display results
        print(f"\n== Diseases with top 2 symptom match counts ==")
        all_top_diseases_list = []
        for count in sorted(grouped_diseases.keys(), reverse=True):
            print(f"\n--- Diseases with {count}/{total_symptoms} symptom matches ---")
            for disease_id, data in grouped_diseases[count]:
                print(f"ID: {data['index']} | Disease: {data['name']} | Matches: {data['count']}/{total_symptoms}")
                all_top_diseases_list.append(data['name'])
        
        return {
            'top_diseases': top_diseases,
            'top_match_counts': top_match_counts,
            'grouped_diseases': dict(grouped_diseases),
            'total_symptoms': total_symptoms,
            'top_diseases_list': all_top_diseases_list
        }
    
no_rag_template = """
You are a medical knowledge assistant specializing in rare diseases. Your task is to provide a differential diagnosis for the following list of symptoms.
List of symptoms: {query_str}

CRITICAL INSTRUCTIONS:
1. Use the information from the context and your own knowledge to provide a comprehensive answer.
2. Return maximum the 10 most relevant diseases, ordered by relevance.
3. Use medical terminology to refer to the diseases, without abreviations.
4. Return EXACTLY this JSON format:

Always format your response as a VALID JSON:
    {
        "symptoms": [{query_str}],
        "differential_diagnosis": [
            "disease1",
            "disease2",
            ... and so on
        ]
    }

    Do NOT use nested objects. Use exactly "disease" and "symptoms" as shown.
"""

rag_template = """
You are a medical knowledge assistant specializing in rare diseases. Your task is to provide a differential diagnosis for the following list of symptoms.
List of symptoms: {query_str}

Use the following candidate diseases to guide your answer: {text_chunks}

CRITICAL INSTRUCTIONS:
1. Use the information from the context and your own knowledge to provide a comprehensive differential diagnosis.
2. Return maximum the 10 most relevant diseases, ordered by relevance.
3. Use medical terminology to refer to the diseases, without abreviations.
4. Return EXACTLY this JSON format:

Always format your response as a VALID JSON:
    {
        "symptoms": [{query_str}],
        "differential_diagnosis": [
            "disease1",
            "disease2",
            ... and so on
        ]
    }

    Do NOT use nested objects. Use exactly "disease" and "symptoms" as shown.
"""

class Output(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptoms provided by the user")
    differential_diagnosis: List[str] = Field(..., description="List of diseases identified as differential diagnosis") 
output_parser = PydanticOutputParser(Output)

# Initialize dic to store results
rag_results = []
no_rag_results = []

# Ensure all necessary containers are created, set up, and started
# services = ['nebula-metad', 'nebula-graphd', 'nebula-storaged']
# for service in services:
#     os.system(f'udocker pull vesoft/{service}:v3')
#     os.system(f'udocker create --name={service} vesoft/{service}:v3')
#     os.system(f'udocker setup --execmode=F1 {service}')
os.system('udocker ps')


n = ng_let(in_container=True)
n.start() # This takes around 5 mins

# Configure connection pool
config = Config()
config.max_connection_pool_size = 10
connection_pool = ConnectionPool()

print("load phenopacket data...")
output_file = os.path.expanduser('~/scratch-llm/data/phenopackets/phenopackets_json.jsonl')
with open(output_file, 'r') as f:
    phenopackets = [json.loads(line) for line in f]

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

        # Create space if not exists
        # session.execute('CREATE SPACE IF NOT EXISTS PrimeKG(vid_type=FIXED_STRING(256))')
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
        from collections import defaultdict, Counter
        def aggregate_differential_diagnoses(all_runs_results):
            """
            Aggregate multiple runs of differential diagnosis results.
            Each disease gets a score based on its position and frequency.
            """
            position_weights = defaultdict(float)
            disease_scores = defaultdict(float)
            
            for run_result in all_runs_results:
                diseases = run_result.get('response', [])
                for position, disease in enumerate(diseases):
                    # Weight: higher positions get higher scores, but diminishing returns
                    weight = 1.0 / (position + 1)  # Position 0 gets weight 1.0, position 1 gets 0.5, etc.
                    disease_scores[disease] += weight
            
            # Sort diseases by their aggregated scores
            sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return top 10 diseases
            return [disease for disease, score in sorted_diseases[:10]]
        
        from tqdm import tqdm
        for p in tqdm(phenopackets[:4090], desc="Processing "):
            print(f" === RAG for {p['id']}: {p['gold']['disease_name']}", flush=True)
            if not p['symptoms']:
                continue

            context = SymptomsMode(vector_store, graph_store).retrieve(p['symptoms'])
            prompt_template = PromptTemplate(rag_template)
            rag_run_results = []
            no_rag_run_results = []

            ## INCLUDE MULTIPLE RUNS
            for run in range(10):
                print(f" == RAG for {p['id']}: {p['gold']['disease_name']} (run {run+1}) ==", flush=True)
                summarizer = TreeSummarize(verbose=True, llm=llm, summary_template=prompt_template)               

                try:
                    rag_response = safe_llm_call(summarizer,
                    query_str=", ".join(p['symptoms']),
                    text_chunks=", ".join([chunk for chunk in context['top_diseases_list']])
                )
                    if rag_response:
                        parsed_response = output_parser.parse(rag_response)
                        rag_run_result = {
                            "run": run + 1,
                            "response": parsed_response.differential_diagnosis
                        }
                        rag_run_results.append(rag_run_result)
                    
                    
                except ValueError as e:
                    print(f"Skipping RAG run {run+1} {p['id']} due to ValueError: {e}")
                    continue

                except ValueError as e:
                    print(f"Error parsing response for symptoms {p['symptoms']}: {e}")
                    rag_response = Output(symptoms=p['symptoms'], differential_diagnosis=[])
                    rag_result = {
                        "id": p['id'],
                        "gold": {
                            "disease_id": p['gold']['disease_id'],
                            "disease_name": p['gold']['disease_name'],
                        },
                        "symptoms": p['symptoms'],
                        "seed": 1234,
                        "response": rag_response.differential_diagnosis
                    }

                print(f" == no RAG for {p['id']}: {p['gold']['disease_name']}", flush=True)
                try:
                    prompt_template = PromptTemplate(no_rag_template)
                    prompt = prompt_template.format(query_str=", ".join(p['symptoms']))
                    response = llm.chat([ChatMessage(role="user", content=prompt)])

                    response_text = response.message.content if hasattr(response, 'message') else str(response)
                    no_rag_response = output_parser.parse(response_text)

                    no_rag_result = {
                        "run": run + 1,
                        "response": no_rag_response.differential_diagnosis
                    }
                    no_rag_run_results.append(no_rag_result)
                    
                except ValueError as e:
                    print(f"Error parsing no RAG response for symptoms {p['symptoms']}: {e}")
                    no_rag_response = Output(symptoms=p['symptoms'], differential_diagnosis=[])
                    no_rag_result = {
                        "id": p['id'],
                        "gold": {
                            "disease_id": p['gold']['disease_id'],
                            "disease_name": p['gold']['disease_name'],
                        },
                        "symptoms": p['symptoms'],
                        "response": no_rag_response.differential_diagnosis
                    }
            # aggregate results from all runs
            if rag_run_results:
                aggregated_rag_response = aggregate_differential_diagnoses(rag_run_results)
                rag_result = {
                    "id": p['id'],
                    "gold": {
                        "disease_id": p['gold']['disease_id'],
                        "disease_name": p['gold']['disease_name'],
                    },
                    "symptoms": p['symptoms'],
                    "seed": 1234,
                    "response": "\n".join([f"{i+1}. {disease}" for i, disease in enumerate(aggregated_rag_response)])
                }
                rag_results.append(rag_result)
            if no_rag_run_results:
                aggregated_no_rag_response = aggregate_differential_diagnoses(no_rag_run_results)
                no_rag_result = {
                    "id": p['id'],
                    "gold": {
                        "disease_id": p['gold']['disease_id'],
                        "disease_name": p['gold']['disease_name'],
                    },
                    "symptoms": p['symptoms'],
                    "seed": 1234,
                    "response": "\n".join([f"{i+1}. {disease}" for i, disease in enumerate(aggregated_no_rag_response)])
                }
            no_rag_results.append(no_rag_result)
        
        results_file = os.path.expanduser('~/scratch-llm/results/symptoms_mode/symptoms_rag_results1.jsonl')
        with open(results_file, 'w') as f:
            for item in rag_results:
               f.write(json.dumps(item) + '\n')

        results_file = os.path.expanduser('~/scratch-llm/results/symptoms_mode/symptoms_no_rag_results1.jsonl')
        with open(results_file, 'w') as f:
            for item in no_rag_results:
                f.write(json.dumps(item) + '\n')
finally:
    # Release the session and close the connection pool for NebulaGraph
    session.release()
    connection_pool.close()

