from langchain_core.prompts import ChatPromptTemplate
from openrouter import ChatOpenRouter

model = ChatOpenRouter(model_name="meta-llama/llama-3.3-8b-instruct:free")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following from English into {language}"),
        ("user", "{text}"),
    ]
)


def main():
    prompt = prompt_template.invoke({"language": "Chinese", "text": "this is a test"})
    print(model.invoke(prompt))


if __name__ == "__main__":
    main()
