import torch, sys, re
import weave
from transformers import AutoTokenizer

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from vector_graph_retriever import VectorGraphRetriever as VectorGraphRetriever
# weave.init("chatbot")

@weave.op()
def initialize_llm():
    print("Loading the language model...")
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", padding_side="left", device_map="auto")    
    if tokenizer.pad_token_id is None: #no <pad> token previously defined, only eos_token
        tokenizer.pad_token = "<|end_of_text|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
     
    # Initialize Hugging Face LLM
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
        is_chat_model=True,
        
    )
    Settings.llm = llm
    Settings.chunk_size = 1024
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return Settings

@weave.op()
def initialize_rag(llm, mode, user_input, vector_store):
    retriever = VectorGraphRetriever(llm, mode, user_input)
    results = retriever.retrieve_from_vector(llm, mode, user_input, vector_store)

    return results

@weave.op()
def main(): 
    prompt_disease = """
        You are a rare diseases specialist.
        Always answer the user's medical-related questions concisely and shortly in the form of an enumerated list of symptom names.
        In case of any doubt, indicate that you don't know the answer. 

        Instructions:
        - You will receive the name of a disease, return a numbered list of its main symptoms.
        - Inlcude only a brief explanations or additional context.

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

        Examples:
        User: What are the most likely diseases for symptoms: fever, fatigue, and cough?
        Response:
        1. Influenza: a viral infection that attacks your respiratory system.
        2. COVID-19: infectious disease caused by the coronavirus SARS-CoV-2.
        3. Pneumonia: inflamation of the air sacs in one or both lungs.   
        4. Common cold: a viral infection of your nose and throat. 
    """

    def call_chat(llm, system_prompt, user_input):
        return llm.chat(
                [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_input)])

    # Ask the user if they want to use tools
    model_selection = None
    while model_selection not in ["rag", "llm"]:
        model_selection = input("Do you want to use the RAG or only the LLM? (rag/llm): ").strip().lower()
        if (model_selection == "exit") or (model_selection == "quit") :
            print("Goodbye!")
            sys.exit()

        elif model_selection == "rag":
            print("\n--- RAG: ---")
            vector_store = VectorGraphRetriever._init_vector_store()
            llm = initialize_llm()

            while True:
                print("Enter 'disease' to ask for the symptoms of a disease.\nEnter 'symptoms' to ask for a list of diseases based on symptoms.")
                print("Type 'exit' or 'quit' to leave.\n")
                mode = input("\033[1;34mMode: \033[0m").strip()
                if (mode.lower() == "exit") or (mode.lower() == "quit"):
                    print("Goodbye!")
                    del llm
                    torch.cuda.empty_cache()
                    break
                
                elif (mode.lower() == "disease"):
                    user_input = input("\033[1;34mEnter a disease name: \033[0m").strip()
                    results = initialize_rag(llm, mode, user_input, vector_store)

                    print("\nVector Query Results:")
                    for i, node_id in enumerate(results.ids):
                        print(f"Node ID: {node_id}, Score: {results.similarities[i]:.4f}")
                    print("\n")
                    # response = call_chat(llm, prompt_disease, user_input, vector_store)
                    # print(response.message.content,"\n")

                elif (mode.lower() == "symptoms"):
                    user_input = input("\033[1;34mEnter a list of symptoms: \033[0m").strip()
                    results = initialize_rag(llm, mode, user_input, vector_store)

                    print("\nVector Query Results:")
                    for node_id, score in results:
                        print(f"Node ID: {node_id}, Score: {score:.4f}") 
                    print("\n")   
                    # response = call_chat(llm, prompt_symptom, user_input, vector_store)
                    # print(response.message.content,"\n")       

        elif model_selection == "llm":
            llm = initialize_llm()  # Initialize the LLM

            print("\n--- LLM: ---")
            while True:
                print("Enter 'disease' to ask for the symptoms of a disease.\nEnter 'symptoms' to ask for a list of diseases based on symptoms.")
                print("Type 'exit' or 'quit' to leave.\n")
                mode = input("\033[1;34mMode: \033[0m").strip()
                if (mode.lower() == "exit") or (mode.lower() == "quit"):
                    print("Goodbye!")
                    del llm
                    torch.cuda.empty_cache()
                    break
                
                elif (mode.lower() == "disease"):
                    user_input = input("\033[1;34mEnter a disease name: \033[0m").strip()
                    response = call_chat(llm, prompt_disease, user_input)
                    print(response.message.content,"\n")

                elif (mode.lower() == "symptoms"):
                    user_input = input("\033[1;34mEnter a list of symptoms: \033[0m").strip()
                    response = call_chat(llm, prompt_symptom, user_input)
                    print(response.message.content,"\n")
            
if __name__ == "__main__":
    main()
