import torch, sys, time, threading
import weave
from transformers import AutoTokenizer

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from vector_graph_retriever import VectorGraphRetriever as VectorGraphRetriever

@weave.op()
def initialize_llm():
    import os, io
    from contextlib import redirect_stdout, redirect_stderr
    
    # Suppress progress bars by setting environment variables
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_OFFLINE"] = "1"  # Prevent online checks
    
    # Add sleep intervals to make loading less intensive
    def throttled_loading():
        from llama_index.llms.huggingface import HuggingFaceLLM
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", 
                                                padding_side="left", 
                                                device_map="auto",
                                                verbose=False)    
        time.sleep(0.1)  # Release CPU briefly
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = "<|end_of_text|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        # Initialize Hugging Face LLM with throttling
        llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=1024,
            generate_kwargs={
                "temperature": 0.65, 
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "top_k": 5, 
                "top_p": 0.85
            },
            tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            stopping_ids=[tokenizer.eos_token_id],
            tokenizer_kwargs={"max_length": None},
            model_kwargs={"torch_dtype": torch.float16},
            is_chat_model=True
        )
        time.sleep(0.1)  # Release CPU briefly
        
        Settings.llm = llm
        Settings.chunk_size = 1024
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return Settings
    
    # Redirect stdout and stderr to capture all output
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        return throttled_loading()

@weave.op()
def initialize_rag(llm, mode, user_input, vector_store, graph_store):
    retriever = VectorGraphRetriever(llm, mode, user_input, vector_store, graph_store)
    vector_nodes = retriever.retrieve_from_vector(Settings.embed_model, mode, user_input, vector_store)
    results, context = retriever.retrieve_from_graph(graph_store, vector_nodes)
    return results, context

@weave.op()
def main():
    # Create a shared object to store both LLM and vector store
    class SharedState:
        def __init__(self):
            self.llm = None
            self.vector_store = None
            self.vector_store_ready = False
            self.vector_store_initializing = False
            self.graph_store = None
            self.graph_store_ready = False
            self.graph_store_initializing = False
    
    shared_state = SharedState()
    
    def initialize_vector_store_background():
        """Function to run vector store initialization in background thread"""
        shared_state.vector_store_initializing = True
        try:
            shared_state.vector_store = VectorGraphRetriever._init_vector_store()
            shared_state.vector_store_ready = True
        except Exception as e:
            print(f"Error initializing vector store: {e}")
        shared_state.vector_store_initializing = False
        
    def initialize_graph_store_background():
        """Function to run graph store initialization in background thread"""
        shared_state.graph_store_initializing = True
        try:
            shared_state.graph_store = VectorGraphRetriever._init_graph_store()
            # Only mark as ready if we got a valid graph store
            shared_state.graph_store_ready = shared_state.graph_store is not None
            if not shared_state.graph_store_ready:
                print("Graph store initialization failed (returned None)")

        except Exception as e:
            print(f"Error initializing graph store: {e}")
            shared_state.graph_store_ready = False
        shared_state.graph_store_initializing = False
    
    def ensure_vector_store_ready():
        """ Function to ensure vector store is ready when needed"""
        # Start initialization if not already started
        if not shared_state.vector_store_ready and not shared_state.vector_store_initializing:
            print("Starting vector store initialization...")
            vector_thread = threading.Thread(target=initialize_vector_store_background)
            vector_thread.daemon = True
            vector_thread.start()
        
        # Wait for initialization to complete if needed
        if not shared_state.vector_store_ready:
            print("Waiting for vector store to finish initializing...")
            spinner = ["|", "/", "-", "\\"]
            i = 0
            while not shared_state.vector_store_ready:
                i = (i + 1) % 4
                print(f"\rWaiting for vector store {spinner[i]}", end="")
                time.sleep(0.1)
            
    def ensure_graph_store_ready():
        """ Function to ensure graph store is ready when needed"""
        # Start initialization if not already started
        if not shared_state.graph_store_ready and not shared_state.graph_store_initializing:
            print("Starting graph store initialization...")
            graph_thread = threading.Thread(target=initialize_graph_store_background)
            graph_thread.daemon = True
            graph_thread.start()
        
        # Wait for initialization to complete if needed
        if not shared_state.graph_store_ready:
            print("Waiting for graph store to finish initializing...")
            spinner = ["|", "/", "-", "\\"]
            i = 0
            while not shared_state.graph_store_ready:
                i = (i + 1) % 4
                print(f"\rWaiting for graph store {spinner[i]} ", end="")
                time.sleep(0.1)
            print("\rGraph store ready!            ")

    # Start stores initialization in background immediately
    print("Initializing stores in background...")
    vector_thread = threading.Thread(target=initialize_vector_store_background)
    graph_thread = threading.Thread(target=initialize_graph_store_background)
    vector_thread.daemon = True
    graph_thread.daemon = True
    vector_thread.start()
    graph_thread.start()
    
    # Initialize LLM directly (not in background) - this will block until finished
    print("Initializing language model...")
    try:
        settings = initialize_llm()
        shared_state.llm = settings.llm
        print("Language model ready!")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)  # Exit if LLM initialization fails

    # Prompts definition
    prompt_disease = """
        You are a rare diseases specialist.
        Always answer the user's medical-related questions concisely and shortly in the form of an enumerated list of symptoms.
        In case of any doubt, indicate that you don't know the answer. 

        Instructions:
        - You will receive the name of a disease, return a numbered list of its main symptoms.
        - Inlcude only a brief explanations or additional context.
        - In case you receive any context, include if it's relevant to generate a response

        User: What are the main symptoms of pneumonia?
        Response:
        1. Fever: a temporary increase in your body temperature.
        2. Cough: a sudden, often repetitive, protective reflex.
        3. Shortness of breath: a feeling of not being able to get enough air.
        4. Chest pain: discomfort or pain in the chest.
    """
    prompt_symptom = """
        You are a rare diseases specialist.
        Always answer the user's medical-related questions concisely and shortly in the form of an enumerated list of disease names.
        In case of any doubt, indicate that you don't know the answer. 

        Instructions:
        - You will receive a list of symptoms, return a numbered list of possible diseases.
        - Inlcude only a brief explanations or additional context.
        - In case you receive any context, include if it's relevant to generate a response

        Examples:
        User: What are the most likely diseases for symptoms: fever, fatigue, and cough?
        Response:
        1. Influenza: a viral infection that attacks your respiratory system.
        2. COVID-19: infectious disease caused by the coronavirus SARS-CoV-2.
        3. Pneumonia: inflamation of the air sacs in one or both lungs.   
        4. Common cold: a viral infection of your nose and throat. 
    """

    def call_chat(prompt_template, user_input, rag_context=None):
        """Generate a response using the LLM with optional RAG context."""
        if rag_context:
            # Add RAG context to the system prompt
            enhanced_prompt = f"{prompt_template}\n\nRelevant context:\n{rag_context}"
        else:
            enhanced_prompt = prompt_template
            
        response = Settings.llm.chat([
            ChatMessage(role=MessageRole.SYSTEM, content=enhanced_prompt),
            ChatMessage(role=MessageRole.USER, content=user_input)
        ])
        return response

    print("\nApplication ready")
    # Allow user interaction
    while True:
        model_selection = input("\nDo you want to use the RAG or the LLM only? (rag/llm): ").strip().lower()
        
        if (model_selection == "exit") or (model_selection == "quit"):
            print("Goodbye!")
            #close Nebula connection
            torch.cuda.empty_cache()
            VectorGraphRetriever.stop_graph_store()
            sys.exit()

        elif model_selection == "rag":
            print("\n--- RAG: ---")
            
            while True:
                print("\nEnter 'disease' to ask for the symptoms of a disease.")
                print("Enter 'symptoms' to ask for a list of diseases based on symptoms.")
                print("Type 'exit' or 'quit' to leave.\n")
                
                mode = input("\033[1;34mMode: \033[0m").strip()
                
                if (mode.lower() == "exit") or (mode.lower() == "quit"):
                    print("Goodbye, leaving the RAG mode!")
                    break
                    
                elif (mode.lower() == "disease"):
                    user_input = input("\033[1;34mEnter a disease name: \033[0m").strip()
                    ensure_vector_store_ready() # Wait for vector and graph store to be ready
                    ensure_graph_store_ready()
                    results, rag_context = initialize_rag(shared_state.llm, mode, user_input, shared_state.vector_store, shared_state.graph_store)
                    
                    print("\nGraph query Results:")
                    print(f"Nodes associated with {user_input}:")
                    if results:
                        for node in results:                         
                            print(f"  ID: {node.get('node_index', 'Unknown'):<10} | "
                                  f"Score (vector): {node.get('score'):.4f} | "
                                  f"Name: {node.get('node_name', 'Unknown')}")
                    else:
                        print("  No relevant nodes found.")
                    print(f"\nContext retieved: {rag_context[:200]}")
                    response = call_chat(prompt_disease, user_input, rag_context)
                    print(response.message.content,"\n")
                        
                
                elif (mode.lower() == "symptoms"):
                    user_input = input("\033[1;34mEnter a list of symptoms: \033[0m").strip()
                    ensure_vector_store_ready() # Wait for vector and graph store to be ready
                    ensure_graph_store_ready()
                    results, rag_context = initialize_rag(shared_state.llm, mode, user_input, shared_state.vector_store, shared_state.graph_store)

                    if results:
                        for node in results:                         
                            print(f"  ID: {node.get('node_index', 'Unknown'):<10} | "
                                f"Score (vector): {node.get('score'):.4f} | "
                                f"Name: {node.get('node_name', 'Unknown')}")
                    else:
                        print("  No relevant nodes found.")
                    print(f"\nContext retieved: {rag_context[:200]}")
                    response = call_chat(prompt_symptom, user_input, rag_context)
                    print(response.message.content,"\n")

                    
        elif model_selection == "llm":
            print("\n--- LLM: ---")
            
            while True:
                print("Enter 'disease' to ask for the symptoms of a disease.\nEnter 'symptoms' to ask for a list of diseases based on symptoms.")
                print("Type 'exit' or 'quit' to leave.\n")
                mode = input("\033[1;34mMode: \033[0m").strip()
                if (mode.lower() == "exit") or (mode.lower() == "quit"):
                    print("Goodbye, leaving the LLM mode!")
                    break
                
                elif (mode.lower() == "disease"):
                    user_input = input("\033[1;34mEnter a disease name: \033[0m").strip()
                    response = call_chat(prompt_disease, user_input)
                    print(response.message.content,"\n")

                elif (mode.lower() == "symptoms"):
                    user_input = input("\033[1;34mEnter a list of symptoms: \033[0m").strip()
                    response = call_chat(prompt_symptom, user_input)
                    print(response.message.content,"\n")
            
if __name__ == "__main__":
    main()
