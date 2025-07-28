import os, pickle, time, json, signal, torch 
import numpy as np
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebulagraph_lite import nebulagraph_let as ng_let
import pathlib, os, json
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
from llama_index.core.vector_stores.types import MetadataFilters, FilterOperator
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tqdm import tqdm
from typing import List
from numpy import dot
from numpy.linalg import norm
from pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser


print("Loading embeddings...")
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

## LOAD LLM, EMBEDDING MODEL AND TOKENIZER
print("Loading tokenizer and LLM...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", padding_side="left", device_map="auto")    
if tokenizer.pad_token_id is None: #no <pad> token previously defined, only eos_token
    tokenizer.pad_token = "<|end_of_text|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# PASS TO THE NEXT ITEM IF LLM ENTERS AN INFINITE LOOP
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

## DISEASE MODE CLASS
print("Loading disease mode...")
disease_dict = {
    "key": "node_type",
    "value": "disease",
    "operator": FilterOperator.EQ
}
disease_filter = MetadataFilters(filters=[disease_dict])

class DiseaseMode():
    def __init__(self, vector_store: SimpleVectorStore, graph_store: NebulaPropertyGraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve(self, query: str):        
        query_embedding = Settings.embed_model.get_text_embedding(query)
        vector_results = self.vector_store.query(
            VectorStoreQuery(
                query_embedding=query_embedding, 
                similarity_top_k=1,
                filters=disease_filter,
            )
        )
        
        top_node_id = vector_results.ids[0]
        top_node_score = vector_results.similarities[0]
        kg_node = self.graph_store.get(ids=[top_node_id])[0]
                
        results = [{ # Create results list with primary node
            "node_index": kg_node.id_,
            "node_name": kg_node.properties["node_name"],
            "text": getattr(kg_node, "text", ""),
            "score": top_node_score
        }]
        
        print(f"Best node from vector query: Node ID: {kg_node.id_}, "
              f"Score: {top_node_score:.4f}, Name: {kg_node.properties['node_name']}")
        
        # Find related nodes through graph query
        graph_nodes = self.graph_store.structured_query(
            """
            MATCH (e:Node__) WHERE id(e) == $ids
            MATCH p=(e)-[r:Relation__{label:"disease-disease"}]-(t) 
            UNWIND relationships(p) as rel
            RETURN DISTINCT id(t), t.Props__.node_name, t.Chunk__.text
            """, 
            param_map={"ids": top_node_id}
        )
        
        # Calculate similarity for related nodes and add relevant ones to results
        all_similarities = []
        for node in graph_nodes:
            node_text = node["t.Props__.node_name"] + ": " + node["t.Chunk__.text"]
            node_embedding = Settings.embed_model.get_text_embedding(node_text)
                    
            similarity = dot(query_embedding, node_embedding) / (norm(query_embedding) * norm(node_embedding))
            all_similarities.append((node, similarity))
            
        if len(all_similarities) > 3:
            sim = [s for _, s in all_similarities]
            threshold = np.percentile(sim, 75) # keep top 25% of nodes
        else:
            threshold = 0.7
        
        for node, similarity in all_similarities:
            if similarity > threshold and similarity >= 0.5:
                results.append({
                    "node_index": node["id(t)"],
                    "node_name": node["t.Props__.node_name"],
                    "text": node_text,
                    "score": similarity
                })
        
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        print("\nBest related nodes from graph query:")
        for node in results:  # Skip primary node
            print(f"ID: {node['node_index']} | node name: {node['node_name']} | score: {node['score']:.4f}")
        
        graph_phenotype = graph_store.structured_query(
            """
            MATCH (e:Node__) WHERE id(e) == $ids
            MATCH (e)-[r:Relation__{label:"disease-phenotype-positive"}]-(t) 
            RETURN DISTINCT id(t), t.Props__.node_name
            """, 
            param_map={"ids": top_node_id}
        )
        # join the phenotype names without the " '' " characters
        phenotypes = ", ".join(node["t.Props__.node_name"].replace("'", "") for node in graph_phenotype)

        nodes_with_text = [node for node in results if node['text'].strip()]
        context = [f"'{node['node_name']}': {node['text']}" for node in nodes_with_text] if nodes_with_text else None
        phenotype_context = [f"Is associated with the following phenotypes: {phenotypes}\n"] if phenotypes else None
        
        if results:
            return (context, phenotype_context, top_node_id)
        else:
            return [f"No graph relationships found for {results[0]['node_name']}"] if results else ["No results found"]

print("prompt templates...")
context_phenotype_template = """    
    Context information is below:
    {text_chunks}

    Phenotype context is below:
    {phenotype_context}

    You are a medical knowledge assistant specializing in rare diseases. Your task is to create a comprehensive list of symptoms for {query_str}.

    CRITICAL INSTRUCTIONS:
    1. Use the information from the context and your own knowledge to provide a comprehensive answer
    2. Return MAXIMUM the 16 most relevant symptoms, if there are more than 16 symptoms, return the most relevant ones
    3. Use HPO medical terminology and avoid using including redundant symptoms
    4. Return EXACTLY this JSON format (no variations):

    Always format your response as a VALID JSON:
    {
        "disease": "{query_str}",
        "symptoms": [
            "name symptom1 using HPO terminology",
            "name symptom2 using HPO terminology",
            ... and so on
        ]
    }

    Do NOT use nested objects. Use exactly "disease" and "symptoms" as shown.
"""

no_rag_template = """
    You are a medical knowledge assistant specializing in rare diseases. Your task is to create a comprehensive list of symptoms for {query_str}.

    CRITICAL INSTRUCTIONS:
    1. Use only your own knowledge to provide a comprehensive answer
    2. Return MAXIMUM the 16 most relevant symptoms, if there are more than 16 symptoms, return the most relevant ones
    3. Use only HPO medical terminology and avoid including redundant symptoms
    4. Return EXACTLY this JSON format (no variations):

    Always format your response as a VALID JSON:
    {
        "disease": "{query_str}",
        "symptoms": [
            "name symptom1 using HPO terminology",
            "name symptom2 using HPO terminology",
            "name symptom3 using HPO terminology",
            ...
        ]
    }

    Do NOT use nested objects. Use exactly "disease" and "symptoms" as shown.
"""

print("JSON formatting...")
class Output(BaseModel):
    disease: str = Field(description="The disease of interest given by the user.")
    symptoms: List[str] = Field(description="Symptoms associated with the disease.")
output_parser = PydanticOutputParser(Output)

# Format the response from the LLM
def get_prompt_and_inputs(context, phenotypes):
    prompt_template = PromptTemplate(context_phenotype_template)
    if context and phenotypes:
        return prompt_template, "\n".join(context), phenotypes
    elif context and not phenotypes:
        return prompt_template, "\n".join(context), ""
    elif not context and phenotypes:
        return prompt_template, "", phenotypes
    else:
        return None, None, None

print("load phenopacket data...")
output_file = os.path.expanduser('~/scratch-llm/storage/phenopackets/phenopacket_data.json')
with open(output_file, 'r') as f:
    phenopackets = json.load(f)

## INITIALIZE NEBULAGRAPH
# # Ensure all necessary containers are created
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
        
        rag_results = {} # store results
        no_rag_results = {}
        for disease in tqdm(list(phenopackets.keys()), desc="Processing diseases"): # running using the diseases in Phenopackets
            unique_symptoms_rag = set()
            unique_symptoms = set()
            main_node = []
            context, phenotypes, top_node_id = DiseaseMode(vector_store, graph_store).retrieve(disease)
            prompt_template, text_chunks, phenotype_context = get_prompt_and_inputs(context, phenotypes) # nodes retrieved are the same among calls, so just generate answers with the same context

            print("\n\n", flush=True)
            for run in range(10):
                print(f" == RAG: {disease} (Run {run+1}) ==", flush=True)
                if prompt_template:
                    summarizer = TreeSummarize(verbose=True, llm=llm, summary_template=prompt_template)
                    response = safe_llm_call(summarizer,
                        query_str=disease,
                        text_chunks=text_chunks,
                        phenotype_context=phenotype_context,
                        timeout=300  
                    )
                elif response is None:
                    rag_results[disease] = {
                        "top_node_id": top_node_id if 'top_node_id' in locals() else None,
                        "symptoms": []
                    }
                    continue

                else:
                    response = None
                    print(f"No context or phenotype data available for {disease}. Skipping summarization.", flush=True)
                    rag_results[disease] = {
                        "top_node_id": None,
                        "symptoms": []
                    }
                    continue
                try:
                    if response is None:
                        print(f"Warning: LLM response was None for {disease}. Skipping.", flush=True)
                        rag_response = Output(disease=disease, symptoms=[])
                        print("\n", rag_response.model_dump_json())
                        rag_results[disease] = {
                            "top_node_id": top_node_id if 'top_node_id' in locals() else None,
                            "symptoms": []
                        }
                        continue

                    rag_response = output_parser.parse(response)
                    print(f"RESPONSE OK: {rag_response.model_dump_json()}", flush=True)
                    print(f"Symptoms retrieved: {len(rag_response.symptoms)} for {rag_response.disease}\n", flush=True)

                    unique_symptoms_rag.update(rag_response.symptoms)
                    
                    rag_results[disease] = {
                        "top_node_id": top_node_id if 'top_node_id' in locals() else None,
                        "symptoms": list(unique_symptoms_rag)
                    }
                except (RecursionError, ValueError) as e:
                    print(f"Warning ERROR: {e}")
                    rag_response = Output(disease=disease, symptoms=[])
                    print("\n", rag_response.model_dump_json())
                    continue

                print(f" == no RAG: {disease} (Run {run+1}) ==", flush=True)
                template = PromptTemplate(no_rag_template)
                prompt = template.format(query_str=disease)
                response = llm.chat([ChatMessage(role="user", content=prompt)])

                response_text = response.message.content if hasattr(response, 'message') else str(response)
                try:
                    no_rag_response = output_parser.parse(response_text)
                    print("RESPONSE OK:", no_rag_response.model_dump_json())
                    print(f"Symptoms retrieved: {len(no_rag_response.symptoms)} for {no_rag_response.disease}\n")
                    unique_symptoms.update(no_rag_response.symptoms)

                except (RecursionError, ValueError) as e:
                    print(f"Warning ERROR: {e}")
                    # Fallback: create an Output object with empty symptoms and the disease name
                    no_rag_response = Output(disease=disease, symptoms=[])
                    print("\n", no_rag_response.model_dump_json())
                    continue

                no_rag_results[disease] = {
                    "symptoms": list(unique_symptoms)
                }

        results_file = os.path.expanduser('~/scratch-llm/results/disease_mode/disease_rag_results.json')
        with open(results_file, 'w') as f:
           json.dump(rag_results, f, indent=2)

        results_file = os.path.expanduser('~/scratch-llm/results/disease_mode/disease_no_rag_results.json')
        with open(results_file, 'w') as f:
            json.dump(no_rag_results, f, indent=2)

finally:
    session.release() # close Nebulagraph
    connection_pool.close()