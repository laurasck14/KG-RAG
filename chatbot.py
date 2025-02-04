import torch, sys, textwrap
from transformers import pipeline, AutoTokenizer
from transformers.agents import CodeAgent, Tool
from llama_index.llms.huggingface import HuggingFaceLLM

from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage

# Load a pre-trained Hugging Face model

def system_tool_call():
    content = textwrap.dedent("""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in composing functions. You are given a question and a set of possible functions.
        Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
        If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
        also point it out. You should only return the function call in tools call sections.

        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
        You SHOULD NOT include any other text in the response.

        Here is a list of functions in JSON format that you can invoke.
                              
        [
            {
                "name": "get_disease_symptoms",
                "description": "Retrieve the main symptoms of a specified disease",
                "parameters": {
                    "type": "dict",
                    "required": [
                        "disease_name"
                    ],
                    "properties": {
                        "disease_name": {
                            "type": "string",
                            "description": "The name of the disease"
                        }
                    }
                }
            },
            {
                "name": "diagnose_disease",
                "description": "Diagnose diseases based on a list of symptoms",
                "parameters": {
                    "type": "dict",
                    "required": [
                        "symptoms"
                    ],
                    "properties": {
                        "symptoms": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "A list of symptoms"
                        }
                    }
                }
            }
        ]""")
    return content.strip()

def initialize_llm():
    print("Loading the language model...")

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", truncation=True)
    if tokenizer.pad_token_id is None: #no <pad> token previously defined, only eos_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    # Set a chat template for the tokenizer
    # tokenizer.chat_template = """
    #     {% if not add_generation_prompt is defined %}
    #         {% set add_generation_prompt = false %}
    #     {% endif %}
    #     {% for message in messages %}   
    #        {{ message['role'] + ': ' + message['content'] }}
    #     {% endfor %}
    #     {% if add_generation_prompt %}
    #         {{ '<|im_start|>assistant\n' }}
    #     {% endif %}
    # """

    system_prompt = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # Initialize the Hugging Face LLM
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={
            "temperature": 0.65, 
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "top_k": 5, 
            "top_p": 0.85},
        system_prompt=system_prompt,
        # query_wrapper_prompt=query_wrapper_prompt,
        tokenizer=tokenizer,
        model_name="meta-llama/Llama-3.2-3B",
        device_map="auto",
        stopping_ids=None, #[50278, 50279, 50277, 1, 0],
        # tokenizer_kwargs={"max_length": 4096},
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.float32},
        # is_chat_model=True,
    )
    return llm

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
            print("\n--- Question for the LLM ---")
            print("Type 'exit' to quit.\n")

            while True:
                user_input = input("User question: ").strip()
                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break

                if user_input:
                    response = llm.chat(
                        [ChatMessage(role="system", content=system_tool_call()), 
                         ChatMessage(role="user", content=user_input)])
                    print(str(response),"\n")
                    


if __name__ == "__main__":
    main()
