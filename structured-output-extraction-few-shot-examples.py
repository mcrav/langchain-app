import sys
from typing import Optional, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import tool_example_to_messages


from openrouter import ChatOpenRouter

# Using paid model as free model returned inaccurate results
model = ChatOpenRouter(model_name="openai/gpt-4.1")


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


class People(BaseModel):
    people: List[Person]


structured_llm = model.with_structured_output(schema=People)

examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        People(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        People(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]

messages = []

for txt, tool_call in examples:
    if tool_call.people:
        # This final message is optional for some providers
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        MessagesPlaceholder("examples"),
        ("human", "{text}"),
    ]
)


def main():
    text = sys.argv[1]
    prompt = prompt_template.invoke({"text": text, "examples": messages})
    print(structured_llm.invoke(prompt))


if __name__ == "__main__":
    main()
