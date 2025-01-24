import torch, sys
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline  # Updated import
from langchain.agents import initialize_agent, Tool
from langchain.agents.tools import tool

from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent


# Load a pre-trained Hugging Face model
def initialize_llm():
    print("Loading the language model...")

    # Check if a GPU is available
    device = 0 if torch.cuda.is_available() else -1

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", truncation=True)
    if tokenizer.pad_token_id is None: #no <pad> token previously defined, only eos_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    # Initialize the Hugging Face pipeline with device
    llm_pipeline = pipeline("text-generation",
            model="meta-llama/Llama-3.2-3B", 
            max_length=100, 
            tokenizer=tokenizer,
            device=device,
            top_k=2,
    )

    # Return the HuggingFacePipeline object
    return HuggingFacePipeline(pipeline=llm_pipeline)

# Define a simple tool to extend functionality
@tool
def format_response_as_list(response: str) -> str:
    """Format the response as a list."""
    items = response.split('\n')
    formatted_response = "\n".join([f"- {item.strip()}" for item in items if item.strip()])
    return formatted_response

# Initialize an agent with optional tools
def initialize_agent_with_tools(llm):
    tools = [format_response_as_list]  # Add more tools here
    return initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

def main(): 
    # # Ask the user if they want to use tools

    model_selection = None
    while model_selection not in ["rag", "llm"]:
        model_selection = input("Do you want to use the RAG or only the LLM? (rag/llm): ").strip().lower()

        if model_selection == "rag":
            rag = print("Not implemented yet, bye") # Initialize the RAG
            sys.exit()

        elif model_selection == "llm":
            llm = initialize_llm()  # Initialize the LLM
            agent = initialize_agent_with_tools(llm)  # Initialize the agent with tools
    
    print("\n--- Question for the LLM ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User question: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input and llm:
            # Directly query the model
            response = llm.invoke(user_input)

        elif user_input and rag:
            # Use the RAG model
            print("Not implemented yet, bye")   
    
        # if agent:
        #     # Use the agent for response
        #     response = agent.run(user_input)          

        # Access the first element of the response list
        print(f"LLM: {response}\n")

if __name__ == "__main__":
    main()
