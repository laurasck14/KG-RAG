import os, time, subprocess, pickle
from typing import List
from nebulagraph_lite import nebulagraph_let as ng_let
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStoreData, SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class PrimeKG:
    """
    A reusable class for managing the PrimeKG constructed with NebulaGraph services.
    Other scripts can inherit from this class to get NebulaGraph functionality.
    Includes:
        - NebulaPropertyGraphStore for PrimeKG nodes and edges
        - SimpleVectorStore for PrimeKG node embeddings
    """
    
    def __init__(self):
        self.ng_instance = None
        self.connection_pool = None
        self.graph_store = None
        self.vector_store = None
        self.is_running = False
    
    def cleanup_containers(self):
        """Remove all udocker containers"""
        try:
            result = subprocess.run(['udocker', 'ps'], capture_output=True, text=True)
            if result.returncode != 0:
                print("No containers to clean up")
                return
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            container_ids = [line.split()[0] for line in lines if line.strip()]
            
            if container_ids:
                print(f"Removing {len(container_ids)} containers...")
                for container_id in container_ids:
                    os.system(f'udocker rm -f {container_id}')
                print("✓ Containers cleaned up")
            else:
                print("No containers found")
                
        except Exception as e:
            print(f"Error cleaning containers: {e}")

    def setup_containers(self):
        """Setup NebulaGraph containers"""
        time.sleep(20)
        services = ['nebula-metad', 'nebula-storaged', 'nebula-graphd']
        print("Setting up NebulaGraph containers...")
        
        for service in services:
            print(f"  Setting up {service}...")
            os.system(f'udocker pull vesoft/{service}:v3')
            os.system(f'udocker create --name={service} vesoft/{service}:v3')
            os.system(f'udocker setup --execmode=F1 {service}')
        
        print("✓ All containers setup complete")

    def test_connectivity(self):
        """Test if NebulaGraph is accessible"""
        try:
            config = Config()
            config.max_connection_pool_size = 2
            test_pool = ConnectionPool()
            
            if test_pool.init([('127.0.0.1', 9669)], config):
                with test_pool.session_context('root', 'nebula') as session:
                    session.execute('SHOW SPACES;')
                    test_pool.close()
                    return True
        except:
            pass
        
        return False

    def start_services(self, max_retries=2):
        """
        Start NebulaGraph services with automatic restart on failure.
        
        Args:
            max_retries (int): Maximum number of restart attempts
            
        Returns:
            bool: True if services started successfully, False otherwise
        """
        for attempt in range(max_retries + 1):
            try:
                print(f"Starting NebulaGraph services (attempt {attempt + 1})...")
                self.ng_instance = ng_let(in_container=True)
                self.ng_instance.start()
                
                # Check if services started successfully
                try:
                    status = self.ng_instance.check_status()
                    if status is None:
                        if self.test_connectivity():
                            print("✓ NebulaGraph is ready and responsive!")
                            self.is_running = True
                            return True
                        else:
                            print("✗ NebulaGraph setup completed but connectivity failed")
                            raise RuntimeError("Connectivity test failed")
                    else:
                        print(f"NebulaGraph status check returned: {status}")
                        raise RuntimeError("NebulaGraph services not ready")
                        
                except RuntimeError as e:
                    if attempt < max_retries:
                        if "services status exception" in str(e) or "BAD" in str(e):
                            print(f"RuntimeError caught: {e}")
                            print("Services status is BAD, restarting containers...")
                        else:
                            print(f"Other RuntimeError: {e}")
                            print("General RuntimeError, restarting containers...")
                        
                        # Restart containers and try again
                        self.stop_services()
                        self.cleanup_containers()
                        self.setup_containers()
                    else:
                        print(f"Failed after {max_retries} retries: {e}")
                        return False
        
            except Exception as e:
                if attempt < max_retries:
                    if "udocker command failed with return code 1" in str(e):
                        print(f"udocker command failed: {e}")
                        print("udocker error detected, cleaning up and restarting...")
                    else:
                        print(f"General exception caught: {e}")
                        print("Unknown error, attempting restart...")
                    
                    # Clean restart for any exception
                    self.stop_services()
                    self.cleanup_containers()
                    self.setup_containers()
                else:
                    print(f"Failed after {max_retries} retries: {e}")
                    return False
        
        return False

    def get_connection_pool(self, max_pool_size=10):
        """
        Get a NebulaGraph connection pool.
        
        Args:
            max_pool_size (int): Maximum connections in pool
            
        Returns:
            ConnectionPool: Initialized connection pool or None if failed
        """
        if not self.is_running:
            print("NebulaGraph services not running. Call start_services() first.")
            return None
        
        try:
            config = Config()
            config.max_connection_pool_size = max_pool_size
            self.connection_pool = ConnectionPool()
            
            if self.connection_pool.init([('127.0.0.1', 9669)], config):
                print("Connection pool initialized successfully")
                return self.connection_pool
            else:
                print("Failed to initialize connection pool")
                return None
                
        except Exception as e:
            print(f"Error creating connection pool: {e}")
            return None

    def get_graph_store(self):
        """
        Get a NebulaPropertyGraphStore instance.
            
        Returns:
            NebulaPropertyGraphStore: Graph store instance or None if failed
        """
        if not self.is_running:
            print("NebulaGraph services not running. Call start_services() first.")
            return None
        
        try:
            self.graph_store = NebulaPropertyGraphStore(
                space="PrimeKG",
                username="root",
                password="nebula",
                url="nebula://localhost:9669",
                props_schema="""`node_index` STRING, `node_type` STRING, `node_id` STRING, `node_name` STRING, 
                    `node_source` STRING, `mondo_id` STRING, `mondo_name` STRING, `group_id_bert` STRING, 
                    `group_name_bert` STRING, `orphanet_prevalence` STRING, `display_relation` STRING """,
            )
            print("Graph store initialized successfully")
            return self.graph_store
            
        except Exception as e:
            print(f"Error creating graph store: {e}")
            return None
        
    def get_vector_store(self):
        """
        Get a SimpleVectorStore instance for node embeddings.
        
        Returns:
            SimpleVectorStore: Vector store instance or None if failed
        """
        if not self.is_running:
            print("NebulaGraph services not running. Call start_services() first.")
            return None
        
        try:
            with open(os.path.expanduser('~/scratch-llm/storage/nodes/all_nodes_all-mpnet-base-v2.pkl'), 'rb') as f:
                all_nodes_embedded: List[TextNode] = pickle.load(f)
            # Initialize the embedding model and create dictionaries from the nodes
            Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

            embedding_dict = {node.id_: node.get_embedding() for node in all_nodes_embedded}
            text_id_to_ref_doc_id = {node.id_: node.ref_doc_id or "None" for node in all_nodes_embedded}
            metadata_dict = {node.id_: node.metadata for node in all_nodes_embedded}

            # Initialize the SimpleVectorStore with the dictionaries
            self.vector_store = SimpleVectorStore(
                data = SimpleVectorStoreData(
                    embedding_dict=embedding_dict,
                    text_id_to_ref_doc_id=text_id_to_ref_doc_id,
                    metadata_dict=metadata_dict,
                ),
                stores_text=True
            )
            print("Vector store initialized successfully")
            return self.vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def stop_services(self):
        """Stop NebulaGraph services and cleanup"""
        try:
            if self.connection_pool:
                self.connection_pool.close()
                print("Connection pool closed")
            
            if self.ng_instance:
                print("Stopping NebulaGraph services...")
                self.ng_instance.stop()
                print("Services stopped")
            
            self.is_running = False
            
        except Exception as e:
            print(f"Error stopping services: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always cleanup"""
        self.stop_services()


# Example usage and testing
def main():
    """Test the NebulaGraph manager"""
    with PrimeKG() as primekg:
        # Start services
        if primekg.start_services():
            print("✓ NebulaGraph is ready!")
            
            # Get connection pool
            pool = primekg.get_connection_pool()
            if pool:
                # Test query
                with pool.session_context('root', 'nebula') as session:
                    result = session.execute('SHOW SPACES;')
                    print(f"Query result: {result}")

                    graph_store = primekg.get_graph_store()
                    vector_store = primekg.get_vector_store()

        else:
            print("✗ Failed to start NebulaGraph")

if __name__ == "__main__":
    main()