import os, json, signal, torch, argparse 
import numpy as np
import pathlib

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

from tqdm import tqdm
from typing import List
from numpy import dot
from numpy.linalg import norm
from pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from src.PrimeKG import PrimeKG

class DiseaseModeGenerator(PrimeKG):
    """
    Disease Mode Generator class that inherits from PrimeKG.
    Generates disease symptoms using RAG and No-RAG approaches.
    """
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self.tokenizer = None
        self.disease_filter = None
        self.output_parser = None
        self.dataset = None
        self.dataset_name = None
        
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

        # Setup disease filter
        disease_dict = {
            "key": "node_type",
            "value": "disease",
            "operator": FilterOperator.EQ
        }
        self.disease_filter = MetadataFilters(filters=[disease_dict])
        
        # Setup output parser
        class Output(BaseModel):
            disease: str = Field(description="The disease of interest given by the user.")
            symptoms: List[str] = Field(description="Symptoms associated with the disease.")
        
        self.output_parser = PydanticOutputParser(Output)
        print("✓ Models setup complete")

    def load_data(self, dataset_file):
        """Load dataset"""
        self.dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        with open(dataset_file, 'r') as f:
            self.dataset = json.load(f)
        print(f"✓ Loaded dataset: {len(self.dataset)} | name: {self.dataset_name}")

    def safe_llm_call(self, summarizer, *args, timeout=300, **kwargs):
        """Safe LLM call with timeout handling"""
        # PASS TO THE NEXT ITEM IF LLM ENTERS AN INFINITE LOOP
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

    def retrieve_disease_context(self, query: str):
        """Retrieve disease context using vector and graph stores"""
        if not self.vector_store or not self.graph_store:
            raise ValueError("Vector store or graph store not initialized")
            
        query_embedding = Settings.embed_model.get_text_embedding(query)
        vector_results = self.vector_store.query(
            VectorStoreQuery(
                query_embedding=query_embedding, 
                similarity_top_k=1,
                filters=self.disease_filter,
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
        # print("\nBest related nodes from graph query:")
        # for node in results: 
        #     print(f"ID: {node['node_index']} | node name: {node['node_name']} | score: {node['score']:.4f}")

        graph_phenotype = self.graph_store.structured_query(
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

    def get_prompt_and_inputs(self, context, phenotypes):
        """Format prompt and inputs for RAG"""
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
        
        prompt_template = PromptTemplate(context_phenotype_template)
        if context and phenotypes:
            return prompt_template, "\n".join(context), phenotypes
        elif context and not phenotypes:
            return prompt_template, "\n".join(context), ""
        elif not context and phenotypes:
            return prompt_template, "", phenotypes
        else:
            return None, None, None

    def generate_symptoms(self, outdir=None, runs=5):
        """
        Generate symptoms for all diseases using RAG and No-RAG approaches
        
        Args:
            outdir (str): Output directory for results
            runs (int): Number of runs per disease
        """
        if outdir is None:
            outdir = os.path.expanduser('home/lasa14/scratch-llm/results/disease_mode/')

        os.makedirs(outdir, exist_ok=True)

        if not self.dataset:
            raise ValueError("Phenopackets not loaded. Call load_data() first.")
        
        if not self.vector_store or not self.graph_store:
            raise ValueError("Stores not initialized. Call start_services() and get stores first.")

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

        rag_results = {}
        no_rag_results = {}

        
        for disease in tqdm(list(self.dataset.keys()), desc="Processing diseases"):
            unique_symptoms_rag = set()
            unique_symptoms = set()

            no_rag_results[disease] = {
                "symptoms": []  # Initialize empty list for no-RAG results
            }
            
            context, phenotypes, top_node_id = self.retrieve_disease_context(disease)
            prompt_template, text_chunks, phenotype_context = self.get_prompt_and_inputs(context, phenotypes)
            for run in range(runs):
                print(f" == RAG: {disease} (Run {run+1}) ==", flush=True)
                if prompt_template and (text_chunks is not None or phenotype_context is not None):
                    summarizer = TreeSummarize(verbose=True, llm=self.llm, summary_template=prompt_template)
                    response = self.safe_llm_call(summarizer,
                        query_str=disease,
                        text_chunks=text_chunks,
                        phenotype_context=phenotype_context,
                        timeout=400  
                    )

                    if response is None:  # LLM call failed
                        rag_results[disease] = {
                            "top_node_id": top_node_id if 'top_node_id' in locals() else None,
                            "symptoms": []
                        }
                    else:  # there is a response
                        try:    
                            rag_response = self.output_parser.parse(response)
                            print("RESPONSE OK", flush=True)
                            unique_symptoms_rag.update(rag_response.symptoms)

                            rag_results[disease] = {
                                "top_node_id": top_node_id if 'top_node_id' in locals() else None,
                                "symptoms": list(unique_symptoms_rag)
                            }
                        except (RecursionError, ValueError) as e:  # response has formatting errors
                            print(f"RAG failed for {disease}, skipping this run.", flush=True)

                else:  # no context or phenotype data available
                    print(f"No context or phenotype data available for {disease}. Skipping summarization.", flush=True)
                    rag_results[disease] = {
                        "top_node_id": top_node_id if 'top_node_id' in locals() else None,
                        "symptoms": []
                    }

                print(f" == no RAG: {disease} (Run {run+1}) ==", flush=True)
                template = PromptTemplate(no_rag_template)
                prompt = template.format(query_str=disease)
                response = self.llm.chat([ChatMessage(role="user", content=prompt)])

                response_text = response.message.content if hasattr(response, 'message') else str(response)
                try:
                    no_rag_response = self.output_parser.parse(response_text)
                    print("RESPONSE OK", flush=True)
                    # print(f"Symptoms retrieved: {len(no_rag_response.symptoms)} for {no_rag_response.disease}\n")
                    unique_symptoms.update(no_rag_response.symptoms)

                    no_rag_results[disease] = {
                        "symptoms": list(unique_symptoms)
                    }

                except (RecursionError, ValueError) as e:
                    print(f"No-RAG failed for {disease}, skipping this run.", flush=True)
                print("\n\n", flush=True)


        # Save results
        rag_results_file = os.path.join(outdir, f'{self.dataset_name}_rag_results.json')
        with open(rag_results_file, 'w') as f:
           json.dump(rag_results, f, indent=2)
        print(f"✓ RAG results saved to: {rag_results_file}")

        no_rag_results_file = os.path.join(outdir, f'{self.dataset_name}_no_rag_results.json')
        with open(no_rag_results_file, 'w') as f:
            json.dump(no_rag_results, f, indent=2)
        print(f"✓ No-RAG results saved to: {no_rag_results_file}")
        
        return rag_results, no_rag_results

    def run_disease_mode(self, dataset_file, outdir=None, runs=5):
        """
        Complete disease mode pipeline
        
        Args:
            dataset_file (str): Path to dataset file
            outdir (str): Output directory for results
            runs (int): Number of runs per disease
        """
        try:
            # Load dataset
            self.load_data(dataset_file)

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
                
                # Generate symptoms
                rag_results, no_rag_results = self.generate_symptoms(outdir, runs)
                
                print(f"✓ Disease mode generation complete!")
                print(f"  Processed {len(rag_results)} diseases with RAG")
                print(f"  Processed {len(no_rag_results)} diseases without RAG")
                
                return rag_results, no_rag_results
                
        except Exception as e:
            print(f"Error during disease mode generation: {e}")
            raise
        finally:
            # Always cleanup
            self.stop_services()


def main():
    parser = argparse.ArgumentParser(description="Generate disease symptoms using RAG and No-RAG approaches")
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to the dataset file (json format with disease names as keys)')
    parser.add_argument('--outdir', type=str, 
                       default=os.path.expanduser('/home/lasa14/scratch-llm/results/disease_mode/'),
                       help='Output directory for results')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per disease (default: 5)')
    
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.dataset):
        print(f"Error: The dataset file {args.dataset} does not exist.")
        exit(1)

    # Run disease mode generation
    generator = DiseaseModeGenerator()
    try:
        rag_results, no_rag_results = generator.run_disease_mode(
            args.dataset,
            args.outdir,
            args.runs
        )
        print("✓ Disease mode generation completed successfully!")
        
    except Exception as e:
        print(f"✗ Disease mode generation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()