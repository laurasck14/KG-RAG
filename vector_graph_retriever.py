import os, sys, torch, pickle
from typing import List
from transformers import AutoTokenizer

from llama_index.core import Settings
# from llama_index.core.indices.property_graph import VectorContextRetriever
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
    def __init__(
        self,
        llm: HuggingFaceLLM,
        mode: str, 
        user_input: str,
        vector_store = None, 
        # graph_store = None
        )-> None:

        self.llm = llm
        self.mode = mode
        self.user_input = user_input
        self.vector_store = vector_store
        # self.graph_store = self._init_graph_store()

    @staticmethod
    def _init_vector_store():
        """ Initialize the vector store with the embeddings of all nodes in the graph. """
        print("Loading the vector store... 2 mins")

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
        graph_store = NebulaPropertyGraphStore(
            space= "PrimeKG", 
            username = "root",
            password = "nebula",
            url = "nebula://localhost:9669",
            props_schema= "`node_index` STRING, `node_type` STRING, `node_id` STRING, `node_name` STRING, `node_source` STRING, `mondo_id` STRING, `mondo_name` STRING, `group_id_bert` STRING, `group_name_bert` STRING, `orphanet_prevalence` STRING, `umls_description` STRING, `orphanet_definition` STRING, `orphanet_epidemiology` STRING, `orphanet_clinical_description` STRING, `orphanet_management_and_treatment` STRING, `mayo_symptoms` STRING, `mayo_causes` STRING, `mayo_risk_factors` STRING, `mayo_complications` STRING, `mayo_prevention` STRING, `mayo_see_doc` STRING, `display_relation` STRING, `_node_content` STRING, `_node_type` STRING, `document_id` STRING, `doc_id` STRING, `ref_doc_id` STRING, `triplet_source_id` STRING",
        )
        return graph_store
    
    def _hybrid_vector_query(self, embed_model: HuggingFaceEmbedding, vector_store: SimpleVectorStore, user_input: str):
        """ Query the vector store with a list of symptoms.
            Retrieve nodes based on:
            1. A combined query of all symptoms together
            2. Individual queries for each symptom
            Combine and rank the results based on similarity
        """
        from llama_index.core.vector_stores.simple import VectorStoreQuery
        # Query with all symptoms together
        combined_query = VectorStoreQuery(
            query_embedding=user_input,
            similarity_top_k=5
        )
        combined_results = vector_store.query(combined_query)

        # Query each symptom individually
        individual_results = []
        for symptom in user_input:
            symptom_embedding = embed_model.get_text_embedding(symptom)
            symptom_query = VectorStoreQuery(
                query_embedding=symptom_embedding,
                similarity_top_k=3
            )
            individual_results.extend(vector_store.query(symptom_query).ids)

        # Combine and rank results
        all_results = {}
        for node_id, similarity in zip(combined_results.ids, combined_results.similarities):
            all_results[node_id] = similarity  # Store combined query results

        # Sort by relevance (higher values first)
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:5]  # Return top-k results


    def retrieve_from_vector(self, embed_model: HuggingFaceEmbedding, mode: str, user_input: str, vector_store: SimpleVectorStore): 
        """ Retrieve nodes from the vector store based on the user query.
            The user query can be a disease name or a list of symptoms.
        """
        # Use the already initialized vector store instead of reinitializing
        if mode == "disease":
            print("Retrieving nodes based on disease name...")
            query_embedding = Settings.embed_model.get_text_embedding(user_input)
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=3,
            )
            nodes = vector_store.query(vector_store_query)

        elif mode == "symptoms":
            print("Retrieving nodes based on symptoms...")
            nodes = self._hybrid_vector_query(embed_model, vector_store, user_input)
            
        return nodes
        
    def retrieve_from_graph():
        pass

