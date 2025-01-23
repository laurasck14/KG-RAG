import torch
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline  # Updated import
from langchain.agents import initialize_agent, Tool
from langchain.agents.tools import tool
import sys

# Load a pre-trained Hugging Face model
def initialize_llm():
    print("Loading the language model...")

    # Check if a GPU is available
    device = 0 if torch.cuda.is_available() else -1

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    if tokenizer.pad_token_id is None: #no <pad> token previously defined, only eos_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Initialize the Hugging Face pipeline with device
    llm_pipeline = pipeline("text-generation", 
            model="meta-llama/Llama-3.2-3B", 
            max_length=200, 
            tokenizer=tokenizer,
            device=device)

    # Return the HuggingFacePipeline object
    return HuggingFacePipeline(pipeline=llm_pipeline)

# Define a simple tool to extend functionality
@tool
def repeat_input_tool(query: str) -> str:
    """A tool that repeats the user query."""
    return f"You said: {query}"

# Initialize an agent with optional tools
def initialize_agent_with_tools(llm):
    tools = [repeat_input_tool]  # Add more tools here
    return initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

def main():
    # Ask for the type of application to be used:
    model_selection = None
    while model_selection not in ["rag", "llm"]:
        model_selection = input("Do you want to use the RAG or only the LLM? (rag/llm): ").strip().lower()

        if model_selection == "rag":
            rag = print("Not implemented yet, bye") # Initialize the RAG
            sys.exit()

        elif model_selection == "llm":
            llm = initialize_llm() # Initialize the LLM

    # Ask the user if they want to use tools
    use_agent = input("Do you want to enable tools/agents? (yes/no): ").strip().lower()
    if use_agent == "yes":
        agent = initialize_agent_with_tools(llm)
    else:
        agent = None

    print("\n--- Chat with the LLM ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if agent:
            # Use the agent for response
            response = agent.run(user_input)
        else:
            # Directly query the model
            response = llm.invoke(user_input)

        print(f"LLM: {response.strip()}")

if __name__ == "__main__":
    main()
