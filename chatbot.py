import torch, sys, re
from transformers import pipeline, AutoTokenizer
from transformers.agents import CodeAgent, Tool
from llama_index.llms.huggingface import HuggingFaceLLM

from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llms import ChatMessage, MessageRole

def initialize_llm():
    print("Loading the language model...")

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")    
    if tokenizer.pad_token_id is None: #no <pad> token previously defined, only eos_token
        tokenizer.pad_token = "<|end_of_text|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
       
    # Initialize the Hugging Face LLM
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=1052,
        generate_kwargs={
            "temperature": 0.65, 
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "top_k": 5, 
            "top_p": 0.85
            },
        tokenizer=tokenizer,
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        device_map="auto",
        stopping_ids=[tokenizer.eos_token_id],
        tokenizer_kwargs={"max_length": None},
        model_kwargs={"torch_dtype": torch.float16},
        is_chat_model=True,
    )
    return llm

def initialize_rag():
    # storage_context = StorageContext.from_defaults(persist_dir="~/scratch-llm/storage/PrimeKG_index_2/")
    # index = load_index_from_storage(storage_context)
    # query_engine = index.as_query_engine()
    # return query_engine
    pass

def main(): 
    system_prompt = """
        You are a rare diseases specialist.
        Always answer the user's medical-related questions concisely and shortly in the form of an enumerated list of disease names, without explanation and summarizing the main results. 

        Instructions:
        - If the user provides a list of symptoms, return a numbered list of **possible diseases**.
        - If the user asks for the symptoms of a disease, return a numbered list of its **main symptoms**.
        - Do not include explanations or additional context — just the list.

        Examples:
        User: What are the most likely diseases for symptoms: fever, fatigue, and cough?
        Response:
        1. Influenza
        2. COVID-19
        3. Pneumonia

        User: What are the main symptoms of pneumonia?
        Response:
        1. Fever
        2. Cough
        3. Shortness of breath
        4. Chest pain
    """
    
    # Ask the user if they want to use tools
    model_selection = None
    while model_selection not in ["rag", "llm"]:
        model_selection = input("Do you want to use the RAG or only the LLM? (rag/llm): ").strip().lower()
        if model_selection == "exit":
            print("Goodbye!")
            sys.exit()

        elif model_selection == "rag":
            print("Not implemented yet, bye") # Initialize the RAG
            sys.exit()
            # rag = initialize_rag()  # Initialize the RAG
            # print("\n--- Question for the RAG ---")
            # print("Type 'exit' to quit.\n")
            
            # while True:
            #     user_input = input("\033[1;34mUser question: \033[0m").strip()
            #     if user_input.lower() == "exit":
            #         print("Goodbye!")
            #         break

            #     if user_input:
            #         response = rag.query(user_input)
            #         print(str(response),"\n")

        elif model_selection == "llm":
            llm = initialize_llm()  # Initialize the LLM
            print("\n--- Question for the LLM ---")
            print("Type 'exit' to quit.\n")

            while True:
                user_input = input("\033[1;34mUser question: \033[0m").strip()
                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break
                
                if user_input:
                    response = llm.chat(
                        [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                        ChatMessage(role=MessageRole.USER, content=user_input)])
                    response = re.sub(r"\*\*(.*?)\*\*", r"\033[1m\1\033[0m", str(response)) # Bolden the text
                    print(response,"\n")
                
if __name__ == "__main__":
    main()
