import torch, sys, re
import weave
from transformers import AutoTokenizer

from llama_index.core.llms import ChatMessage, MessageRole
weave.init("chatbot")

@weave.op()
def initialize_llm():

    print("Loading the language model...")
    from llama_index.llms.huggingface import HuggingFaceLLM

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
    return llm


@weave.op()
def initialize_rag():
    from llama_index.core import StorageContext, load_index_from_storage

    # storage_context = StorageContext.from_defaults(persist_dir="~/scratch-llm/storage/PrimeKG_index_2/")
    # index = load_index_from_storage(storage_context)
    # query_engine = index.as_query_engine() #.as_chat_engine()


    # print("\n--- Question for the RAG ---")
            # print("Type 'exit' to quit.\n")
            
            # while True:
            #     user_input = input("\033[1;34mUser question: \033[0m").strip()
            #     if user_input.lower() == "exit":
            #         print("Goodbye!")
            #         del llm
            #         torch.cuda.empty_cache()
            #         break

            #     if user_input:
            #         response = rag.query(user_input)
            # return response
    pass

@weave.op()
def main(): 
    system_prompt = """
        You are a rare diseases specialist.
        Always answer the user's medical-related questions concisely and shortly in the form of an enumerated list of disease/symptom names.
        In case of any doubt, indicate that you don't know the answer. 

        Instructions:
        - If the user provides a list of symptoms, return a numbered list of **possible diseases**.
        - If the user asks for the symptoms of a disease, return a numbered list of its **main symptoms**.
        - Inlcude only a brief explanations or additional context.

        Examples:
        User: What are the most likely diseases for symptoms: fever, fatigue, and cough?
        Response:
        1. Influenza: a viral infection that attacks your respiratory system.
        2. COVID-19: infectious disease caused by the coronavirus SARS-CoV-2.
        3. Pneumonia: inflamation of the air sacs in one or both lungs.   
        4. Common cold: a viral infection of your nose and throat. 

        User: What are the main symptoms of pneumonia?
        Response:
        1. Fever: a temporary increase in your body temperature.
        2. Cough: a sudden, often repetitive, protective reflex.
        3. Shortness of breath: a feeling of not being able to get enough air.
        4. Chest pain: discomfort or pain in the chest.
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
            print("Not implemented yet, bye") # Initialize the RAG
            sys.exit()
            # response = initialize_rag() 
            # print(str(response),"\n")          

        elif model_selection == "llm":
            llm = initialize_llm()  # Initialize the LLM

            print("\n--- Question for the LLM ---")
            print("Type 'exit' or 'quit' to leave.\n")
            while True:
                user_input = input("\033[1;34mUser question: \033[0m").strip()
                if (user_input.lower() == "exit") or (user_input.lower() == "quit"):
                    print("Goodbye!")
                    del llm
                    torch.cuda.empty_cache()
                    break
                
                elif not (user_input.lower() == "exit") and not (user_input.lower() == "quit"):
                    response = call_chat(llm, system_prompt, user_input)
                    print(response,"\n")
            
if __name__ == "__main__":
    main()
