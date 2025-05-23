from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from enum import Enum
from openrouter import ChatOpenRouter


model = ChatOpenRouter(model_name="meta-llama/llama-3.3-8b-instruct:free")

tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
)


class ScaleEnum(int, Enum):
    One = (1,)
    Two = (2,)
    Three = (3,)
    Four = (4,)
    Five = 5


class SentimentEnum(str, Enum):
    Positive = "Positive"
    Negative = "Negative"
    Neutral = "Neutral"


class Classification(BaseModel):
    sentiment: SentimentEnum = Field(description="The sentiment of the text")
    aggressiveness: ScaleEnum = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


structured_llm = model.with_structured_output(Classification)


def main():
    inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
    prompt = tagging_prompt.invoke({"input": inp})
    response = structured_llm.invoke(prompt)
    print(response)


if __name__ == "__main__":
    main()
