import os, pickle
from typing import List

from llama_index.core import Settings
from llama_index.core.vector_stores.simple import SimpleVectorStoreData, SimpleVectorStore, VectorStoreQuery
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.core.schema import TextNode

class VectorGraphRetriever():
    """ A retriever that uses a vector store to retrieve nodes based on a user query.
        The user query can be a disease name or a list of symptoms.
        The retrieved nodes from the vector store are then used to query the graph store.
    """
    # Class-level variable to store the NebulaGraph instance
    nebula_instance = None

    def __init__(
        self,
        llm: HuggingFaceLLM,
        mode: str, 
        user_input: str,
        vector_store = None, 
        graph_store = None
        )-> None:

        self.llm = llm
        self.mode = mode
        self.user_input = user_input
        self.vector_store = vector_store
        self.graph_store = graph_store

    @staticmethod
    def _init_vector_store():
        """ Initialize the vector store with the embeddings of all nodes in the graph. """
        # print("Loading the vector store... 2 mins")

        with open(os.path.expanduser('~/scratch-llm/storage/all_nodes_embedded.pkl'), 'rb') as f:
            all_nodes_embedded: List[TextNode] = pickle.load(f)
        embedding_dict = {node.id_: node.get_embedding() for node in all_nodes_embedded}
        text_id_to_ref_doc_id = {node.id_: node.ref_doc_id or "None" for node in all_nodes_embedded}
        metadata_dict = {node.id_: node.metadata for node in all_nodes_embedded}

        # Initialize the SimpleVectorStore with the dictionaries
        vector_store = SimpleVectorStore(
            data=SimpleVectorStoreData(
                embedding_dict=embedding_dict,
                text_id_to_ref_doc_id=text_id_to_ref_doc_id,
                metadata_dict=metadata_dict,
            ),
            stores_text=True
        )
        return vector_store
    
    @staticmethod
    def _init_graph_store():
        """ Initialize the graph store with the schema of the graph. """
        from nebulagraph_lite import nebulagraph_let as ng_let
        from nebula3.gclient.net import ConnectionPool
        from nebula3.Config import Config
        import subprocess

        # Try to initialize and start NebulaGraph
        try:
            # Store the instance at class level so we can access it later for shutdown
            VectorGraphRetriever.nebula_instance = ng_let(in_container=True)
            VectorGraphRetriever.nebula_instance.start()  # This takes around 5 mins
        except Exception as e:
            print(f"Error starting NebulaGraph: {e}")
            print("Retrying with manual container setup...")
            
            # Retry with manual container setup
            try:
                services = ['nebula-storaged', 'nebula-metad', 'nebula-graphd']
                for service in services:
                    print(f" Setting up {service}...")
                    # Silent execution with status check
                    try:
                        subprocess.run(f'udocker pull vesoft/{service}:v3', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        subprocess.run(f'udocker create --name={service} vesoft/{service}:v3', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        subprocess.run(f'udocker setup --execmode=F1 {service}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Command failed: {e}")
                
                # Try starting NebulaGraph again
                print(" Retrying NebulaGraph start...")
                VectorGraphRetriever.nebula_instance = ng_let(in_container=True)
                VectorGraphRetriever.nebula_instance.start()
            except Exception as e2:
                print(f"Failed to initialize NebulaGraph after retry: {e2}")
                return None

        # Configure connection pool
        config = Config()
        config.max_connection_pool_size = 10
        connection_pool = ConnectionPool()
        graph_store = None  # Initialize to None

        # Initialize connection pool
        try:
            if not connection_pool.init([('127.0.0.1', 9669)], config):
                raise Exception("Failed to initialize connection pool")
            print("Connection pool initialized successfully")
        except Exception as e:
            print(f"Error initializing connection pool: {e}")
            return None  # Return None instead of exiting

        # Create a session and execute a query
        try:
            with connection_pool.session_context('root', 'nebula') as session:        
                graph_store = NebulaPropertyGraphStore(
                    space= "PrimeKG", 
                    username = "root",
                    password = "nebula",
                    url = "nebula://localhost:9669",
                    props_schema= "`node_index` STRING, `node_type` STRING, `node_idx` STRING, `node_name` STRING, `node_source` STRING, `mondo_id` STRING, `mondo_name` STRING, `group_id_bert` STRING, `group_name_bert` STRING, `orphanet_prevalence` STRING, `umls_description` STRING, `orphanet_definition` STRING, `orphanet_epidemiology` STRING, `orphanet_clinical_description` STRING, `orphanet_management_and_treatment` STRING, `mayo_symptoms` STRING, `mayo_causes` STRING, `mayo_risk_factors` STRING, `mayo_complications` STRING, `mayo_prevention` STRING, `mayo_see_doc` STRING, `display_relation` STRING, `_node_content` STRING, `_node_type` STRING, `document_id` STRING, `doc_id` STRING, `ref_doc_id` STRING, `triplet_source_id` STRING",
                )
        except Exception as e:
            print(f"Error creating graph store: {e}")
            return None
        
        return graph_store
    
    @staticmethod
    def stop_graph_store():
        """Stop the NebulaGraph instance if it exists"""
        if VectorGraphRetriever.nebula_instance is not None:
            try:
                print("Stopping NebulaGraph instance...")
                VectorGraphRetriever.nebula_instance.stop()
                VectorGraphRetriever.nebula_instance = None
                print("NebulaGraph stopped successfully")
                return True
            except Exception as e:
                print(f"Error stopping NebulaGraph: {e}")
                return False
        return False
    
    def _hybrid_vector_query(self, embed_model: HuggingFaceEmbedding, vector_store: SimpleVectorStore, user_input: str):
        """Query the vector store with a list of symptoms."""
        # Parse user input into separate symptoms if needed
        if isinstance(user_input, str):
            # Check if input looks like a JSON array or comma-separated list
            if user_input.strip().startswith("[") and user_input.strip().endswith("]"):
                try:
                    import json
                    symptoms = json.loads(user_input)
                except:
                    # If JSON parsing fails, try comma-separated
                    symptoms = [s.strip() for s in user_input.strip("[]").split(",")]
            else:
                # Assume comma-separated values
                symptoms = [s.strip() for s in user_input.split(",")]
        else:
            symptoms = user_input  # Already a list
        
        print(f"Processing {len(symptoms)} symptoms: {symptoms}")
        
        # Query with all symptoms together - use the combined text for embedding
        combined_text = ", ".join(symptoms)
        combined_embedding = embed_model.get_text_embedding(combined_text)
        combined_query = VectorStoreQuery(
            query_embedding=combined_embedding,
            similarity_top_k=5
        )
        combined_results = vector_store.query(combined_query)

        # Query each symptom individually
        individual_results = []
        for symptom in symptoms:
            symptom_embedding = embed_model.get_text_embedding(symptom)
            symptom_query = VectorStoreQuery(
                query_embedding=symptom_embedding,
                similarity_top_k=3
            )
            individual_results.extend(vector_store.query(symptom_query).ids)

        # Combine and rank results
        all_results = {}
        for node_idx, similarity in zip(combined_results.ids, combined_results.similarities):
            all_results[node_idx] = similarity  # Store combined query results

        # Sort by relevance (higher values first)
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:5]  # Return top-k results


    def retrieve_from_vector(self, embed_model: HuggingFaceEmbedding, mode: str, user_input: str, vector_store: SimpleVectorStore): 
        """ Retrieve nodes from the vector store based on the user query.
            The user query can be a disease name or a list of symptoms.
            Returns nodes in a consistent format: list of (node_idx, score) tuples
        """
        # Use the already initialized vector store instead of reinitializing
        if mode == "disease":
            print(f"Retrieving relevant nodes for {user_input} ...")
            query_embedding = Settings.embed_model.get_text_embedding(user_input)
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=3,
            )
            query_result = vector_store.query(vector_store_query)
            
            # Convert VectorStoreQueryResult to list of (node_idx, score) to match the format of symptoms mode results
            nodes = [(node_idx, score) for node_idx, score in zip(query_result.ids, query_result.similarities)]

        elif mode == "symptoms":
            nodes = self._hybrid_vector_query(embed_model, vector_store, user_input)
            
        return nodes
        
    def retrieve_from_graph(self, graph_store: NebulaPropertyGraphStore, nodes):
        """ Retrieve nodes from the graph store based on the nodes retrieved from the vector store.
            Expects nodes as list of (node_idx, score) tuples.
        """
        # Check if graph_store is valid
        if graph_store is None:
            print("Graph store is not initialized")
            return nodes  # Return the original vector store results as fallback
        
        try:
            # Process list of (node_idx, score) tuples
            kg_ids = [node_idx for node_idx, _ in nodes]
            scores_dict = {node_idx: score for node_idx, score in nodes}
            kg_nodes = graph_store.get(ids=kg_ids) # Get the nodes from graph store
            
            formatted_results = []
            # Process the returned kg_nodes based on their structure
            if hasattr(kg_nodes, '__iter__'):
                for node in kg_nodes:
                    node_idx = None
                    node_name = None
                    score = 0
                                          
                    # Extract node_name and node_idx from properties (both for ChunkNode and EntityNode)
                    if hasattr(node, 'properties') and 'node_name' in node.properties:
                        node_name = node.properties['node_name']
                        node_idx = node.properties['node_index']
                        score = scores_dict.get(node_idx, 0)
                    
                    formatted_results.append({
                        'node_index': node_idx,
                        'node_name': node_name,
                        'score': score
                    })
            
            return formatted_results if formatted_results else nodes
                
        except Exception as e:
            print(f"Error querying graph store: {e}")
            return nodes  # Return the original vector store results as fallback

