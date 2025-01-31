import torch, sys
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline  # Updated import
from transformers.agents import CodeAgent, Tool

from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
# from llama_index.agent.openai import OpenAIAgent


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

# Define custom tools with the required attributes
class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers"
    inputs = {"a": {"type": "number", "description": "The first number"}, "b": {"type": "number", "description": "The second number"}}
    output_type = "number"

    def forward(self, a: float, b: float) -> float:
        return a * b

class AddTool(Tool):
    name = "add"
    description = "Add two numbers"
    inputs = {"a": {"type": "number", "description": "The first number"}, "b": {"type": "number", "description": "The second number"}}
    output_type = "number"

    def forward(self, a: float, b: float) -> float:
        return a + b

def main(): 
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

        elif model_selection == "llm":
            llm = initialize_llm()  # Initialize the LLM
            # multiply_tool = MultiplyTool()
            # add_tool = AddTool()
            # agent = CodeAgent(tools=[multiply_tool, add_tool], llm_engine=llm, verbose=True)  # Initialize the agent
    
    print("\n--- Question for the LLM ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User question: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input :#and agent:
            # Use the agent to handle the query
            response = llm.invoke(user_input)
            # response = agent.run(user_input)
            if isinstance(response, list):
                response = response[0]  # Ensure response is a string
            print(f"LLM:\n{response}\n")

if __name__ == "__main__":
    main()
