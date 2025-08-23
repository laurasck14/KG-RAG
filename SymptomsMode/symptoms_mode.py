import os, json, signal, torch, argparse, pathlib
from collections import defaultdict
from tqdm import tqdm
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HOME'] = str(pathlib.Path("~/scratch-llm/storage/cache/huggingface/").expanduser().absolute())

from transformers import AutoTokenizer
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.vector_stores.simple import VectorStoreQuery
from llama_index.core.vector_stores.types import MetadataFilters, FilterOperator
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from src.PrimeKG import PrimeKG


class Output(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptoms provided by the user")
    differential_diagnosis: List[str] = Field(..., description="List of diseases identified as differential diagnosis") 


class SymptomsModeGenerator(PrimeKG):
    """
    Symptoms Mode Generator class that inherits from PrimeKG.
    Generates differential diagnosis for given symptoms using RAG and No-RAG approaches.
    """
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self.tokenizer = None
        self.phenotype_filter = None
        self.output_parser = None
        self.dataset = None

    def setup_models(self):
        """Setup LLM, tokenizer, and other models"""
        print("Loading tokenizer and LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct", 
            padding_side="left", 
            device_map="auto"
        )    
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = "<|end_of_text|>"
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        self.llm = HuggingFaceLLM(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            context_window=8192,
            max_new_tokens=3048,
            generate_kwargs={
                "temperature": 0.10, 
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "top_k": 10, 
                "top_p": 0.9,
            },
            model_kwargs={
                "torch_dtype": torch.float16,
            },
            tokenizer=self.tokenizer,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            stopping_ids=[self.tokenizer.eos_token_id],
            tokenizer_kwargs={"max_length": None},
            is_chat_model=True,
        )

        Settings.llm = self.llm
        Settings.chunk_size = 1024
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

        # Setup phenotype filter
        phenotype_dict = {
            "key": "node_type",
            "value": "effect/phenotype",
            "operator": FilterOperator.EQ
        }
        self.phenotype_filter = MetadataFilters(filters=[phenotype_dict])
        
        # Setup output parser
        self.output_parser = PydanticOutputParser(Output)
        print("✓ Models setup complete")

    def load_dataset(self, dataset_file):
        """Load dataset data"""
        with open(dataset_file, 'r') as f:
            self.dataset = [json.loads(line) for line in f]
        print(f"✓ Loaded {len(self.dataset)} records from dataset")

    def safe_llm_call(self, summarizer, *args, timeout=180, **kwargs):
        """Safe LLM call with timeout handling"""
        class TimeoutException(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutException()
        
        signal.signal(signal.SIGALRM, handler)
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

    def retrieve(self, query: List[str]):
        """Retrieve diseases based on symptoms"""
        if not isinstance(query, list):
            return None
            
        disease_counter = {}
        total_symptoms = len(query)
        
        # Collect all diseases and count symptom matches
        for symptom in query:
            query_embedding = Settings.embed_model.get_text_embedding(symptom)
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=1,
                filters=self.phenotype_filter,
            )
            individual_results = self.vector_store.query(vector_store_query)
            
            for node_id, score in zip(individual_results.ids, individual_results.similarities):
                # Get related diseases from graph
                graph_nodes = self.graph_store.structured_query(
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
        
        # Get list of top diseases
        all_top_diseases_list = []
        for count in sorted(grouped_diseases.keys(), reverse=True):
            for disease_id, data in grouped_diseases[count]:
                all_top_diseases_list.append(data['name'])
        
        return {
            'top_diseases': top_diseases,
            'top_match_counts': top_match_counts,
            'grouped_diseases': dict(grouped_diseases),
            'total_symptoms': total_symptoms,
            'top_diseases_list': all_top_diseases_list
        }

    def aggregate_differential_diagnoses(self, all_runs_results):
        """
        Aggregate multiple runs of differential diagnosis results.
        Each disease gets a score based on its position and frequency.
        """
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

    def generate_differential_diagnosis(self, runs, outdir=None):
        """
        Generate differential diagnosis for all items in the dataset using RAG and No-RAG approaches

        Args:
            runs (int): Number of runs per item in the dataset
            outdir (str): Output directory for results
        """
        if outdir is None:
            outdir = os.path.expanduser('~/scratch-llm/results/symptoms_mode/')

        os.makedirs(outdir, exist_ok=True)

        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        if not self.vector_store or not self.graph_store:
            raise ValueError("Stores not initialized. Call start_services() and get stores first.")

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

        rag_results = []
        no_rag_results = []

        for p in tqdm(self.dataset, desc="Processing item"):
            print(f"\n\n === {p['id']}: {p['gold']['disease_name']} ===", flush=True)
            if not p['symptoms']:
                print(f"Skipping {p['id']}: {p['gold']['disease_name']} due to no symptoms provided.")
                continue

            context = self.retrieve(p['symptoms'])
            prompt_template = PromptTemplate(rag_template)
            rag_run_results = []
            no_rag_run_results = []

            # Multiple runs
            for run in range(runs):
                print(f" == RAG for {p['id']}: {p['gold']['disease_name']} (run {run+1}) ==", flush=True)

                try:
                    prompt = prompt_template.format(
                        query_str=", ".join(p['symptoms']),
                        text_chunks=", ".join(context['top_diseases_list'])
                    )
                    response = self.llm.chat([ChatMessage(role="user", content=prompt)])
                    response_text = response.message.content if hasattr(response, 'message') else str(response)
                    rag_response = self.output_parser.parse(response_text)
                    rag_run_result = {
                        "run": run + 1,
                        "response": rag_response.differential_diagnosis
                    }
                    rag_run_results.append(rag_run_result)                    
                    
                except ValueError as e:  # context is too large for the LLM
                    try: 
                        summarizer = TreeSummarize(verbose=True, llm=self.llm, summary_template=prompt_template)    
                        rag_response = self.safe_llm_call(summarizer,
                            query_str=", ".join(p['symptoms']),
                            text_chunks=", ".join([chunk for chunk in context['top_diseases_list']])
                        )
                        if not rag_response or not rag_response.differential_diagnosis:
                            print(f"Empty response for symptoms {p['id']}: {p['gold']['disease_name']}")
                            continue

                        rag_run_result = {
                            "run": run + 1,
                            "response": rag_response.differential_diagnosis
                        }
                        rag_run_results.append(rag_run_result)

                    except ValueError as e:  # another error parsing the response
                        print(f"Error parsing response for symptoms {p['symptoms']}: {e}")
                        continue

                print(f" == no RAG for {p['id']}: {p['gold']['disease_name']} (run {run+1}) ==", flush=True)
                try:
                    prompt_template_no_rag = PromptTemplate(no_rag_template)
                    prompt = prompt_template_no_rag.format(query_str=", ".join(p['symptoms']))
                    response = self.llm.chat([ChatMessage(role="user", content=prompt)])

                    response_text = response.message.content if hasattr(response, 'message') else str(response)
                    no_rag_response = self.output_parser.parse(response_text)
                    if not no_rag_response.differential_diagnosis:
                        print(f"Empty no RAG response for symptoms {p['id']}: {p['gold']['disease_name']}")
                        continue
                    
                    no_rag_result = {
                        "run": run + 1,
                        "response": no_rag_response.differential_diagnosis
                    }
                    no_rag_run_results.append(no_rag_result)
                    
                except ValueError as e:
                    print(f"Error parsing no RAG response for symptoms {p['symptoms']}: {e}")
                    continue

            # aggregate results from all runs
            if rag_run_results:
                aggregated_rag_response = self.aggregate_differential_diagnoses(rag_run_results)
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
                aggregated_no_rag_response = self.aggregate_differential_diagnoses(no_rag_run_results)
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

        # Save results
        rag_results_file = os.path.join(outdir, 'symptoms_rag_results.jsonl')
        with open(rag_results_file, 'w') as f:
            for item in rag_results:
               f.write(json.dumps(item) + '\n')
        print(f"✓ RAG results saved to: {rag_results_file}")

        no_rag_results_file = os.path.join(outdir, 'symptoms_no_rag_results.jsonl')
        with open(no_rag_results_file, 'w') as f:
            for item in no_rag_results:
                f.write(json.dumps(item) + '\n')
        print(f"✓ No-RAG results saved to: {no_rag_results_file}")
        
        return rag_results, no_rag_results

    def run_symptoms_mode(self, dataset_file, runs, outdir=None):
        """
        Complete symptoms mode pipeline
        
        Args:
            dataset_file (str): Path to dataset file
            runs (int): Number of runs per item in the dataset
            outdir (str): Output directory for results
        """
        try:
            # Load dataset
            self.load_dataset(dataset_file)

            # Setup models
            self.setup_models()
            
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
                
                # Generate differential diagnosis
                rag_results, no_rag_results = self.generate_differential_diagnosis(outdir, runs)
                
                print(f"✓ Symptoms mode generation complete!")
                print(f"  Processed {len(rag_results)} items with RAG")
                print(f"  Processed {len(no_rag_results)} items without RAG")

                return rag_results, no_rag_results
                
        except Exception as e:
            print(f"Error during symptoms mode generation: {e}")
            raise
        finally:
            # Always cleanup
            self.stop_services()


def main():
    parser = argparse.ArgumentParser(description="Generate differential diagnosis for symptoms using RAG and No-RAG approaches")
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to the dataset file (jsonl format)')
    parser.add_argument('--outdir', type=str, 
                       default=os.path.expanduser('/home/lasa14/scratch-llm/results/symptoms_mode/'),
                       help='Output directory for results')
    parser.add_argument('--runs', type=int, required=True,
                       help='Number of runs per item in the dataset')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.dataset):
        print(f"Error: The dataset file {args.dataset} does not exist.")
        exit(1)

    # Run symptoms mode generation
    generator = SymptomsModeGenerator()
    try:
        rag_results, no_rag_results = generator.run_symptoms_mode(
            args.dataset,
            args.outdir,
            args.runs,
        )
        print("✓ Symptoms mode generation completed successfully!")
        
    except Exception as e:
        print(f"✗ Symptoms mode generation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()

