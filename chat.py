from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import sys
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langchain.schema import get_buffer_string
from tiktoken import encoding_for_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")

workflow = StateGraph(state_schema=MessagesState)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an energy vampire. Do everything you can to drain the user's energy while sounding upbeat yourself.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def count_tokens(messages):
    encoding = encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(get_buffer_string(messages)))


# Limit messages to 65 tokens to avoid overloading context window
trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=count_tokens,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


# Define the function that calls the model
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    response = model.invoke(prompt)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Thread ID allows same app to have multiple conversations at once with different memories per conversation
config = RunnableConfig(configurable={"thread_id": "1"})


def stream_message(message):
    for chunk, metadata in app.stream(
        {"messages": [HumanMessage(content=message)]}, config, stream_mode="messages"
    ):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="|")
    print()


def main():
    stream_message("Hi! I'm Bob")
    stream_message("What's my name?")


if __name__ == "__main__":
    main()
