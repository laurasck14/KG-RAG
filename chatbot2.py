import asyncio
import os
import torch
from transformers import pipeline, AutoTokenizer
from typing import Any, List

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput, FunctionTool
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from langchain_huggingface import HuggingFacePipeline  # Updated import

class CustomHuggingFacePipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.metadata = type('Metadata', (object,), {'context_window': 2048})

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

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
            tokenizer=tokenizer,
            device=device,
            top_k=2,
            max_new_tokens=110,  # Set max_new_tokens instead of max_length
    )

    # Return the CustomHuggingFacePipeline object
    return CustomHuggingFacePipeline(pipeline=llm_pipeline)

class PrepEvent(Event):
    pass

class InputEvent(Event):
    input: list[ChatMessage]

class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]

class FunctionOutputEvent(Event):
    output: ToolOutput

class ReActAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or initialize_llm()
        self.memory = ChatMemoryBuffer.from_defaults(llm=self.llm)
        self.formatter = ReActChatFormatter(context=extra_context or "")
        self.output_parser = ReActOutputParser()
        self.sources = []

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        self.sources = []
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)
        await ctx.set("current_reasoning", [])
        return PrepEvent()

    @step
    async def prepare_chat_history(self, ctx: Context, ev: PrepEvent) -> InputEvent:
        chat_history = self.memory.get()
        current_reasoning = await ctx.get("current_reasoning", default=[])
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> ToolCallEvent | StopEvent:
        chat_history = ev.input
        chat_history_str = " ".join([msg.content for msg in chat_history])
        response = self.llm(chat_history_str)

        try:
            reasoning_step = self.output_parser.parse(response[0]['generated_text'])
            (await ctx.get("current_reasoning", default=[])).append(reasoning_step)
            
            if reasoning_step.is_done:
                self.memory.put(ChatMessage(role="assistant", content=reasoning_step.response))
                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [*self.sources],
                        "reasoning": await ctx.get("current_reasoning", default=[]),
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            (await ctx.get("current_reasoning", default=[])).append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
        return PrepEvent()

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue
            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )
        return PrepEvent()


def add(x: int, y: int) -> int:
    """Useful function to add two numbers."""
    print(f"Adding {x} and {y}")
    return x + y

def multiply(x: int, y: int) -> int:
    """Useful function to multiply two numbers."""
    print(f"Multiplying {x} and {y}")
    return x * y

tools = [
    FunctionTool.from_defaults(add),
    FunctionTool.from_defaults(multiply),
]

agent = ReActAgent(
    llm=initialize_llm(), tools=tools, timeout=120, verbose=True  # Increase the timeout duration
)

async def main():
    user_input = input("You: ")
    ret = await agent.run(input=user_input)
    print(f"Bot: {ret['response']}")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise