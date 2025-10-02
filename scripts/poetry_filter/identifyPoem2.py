import getpass
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
import json


# Define the structured output model
class PoetryOutput(BaseModel):
    page_number: str = Field(description="The page number where poetry was identified, matching the JSON key")
    beginning_words: str = Field(description="The first few words of the poem segment")
    ending_words: str = Field(description="The last few words of the poem segment")


# Initialize Fireworks LLM with API key
if not os.environ.get("FIREWORKS_API_KEY"):
    os.environ["FIREWORKS_API_KEY"] = getpass.getpass("fw_3ZJoNK3svrZn66YSZ7HQaQeq")
llm = init_chat_model("accounts/fireworks/models/kimi-k2-instruct", model_provider="fireworks")

# Set up the output parser
parser = PydanticOutputParser(pydantic_object=PoetryOutput)

# Create a Few-Shot prompt template with escaped JSON
template = """
You are an expert in 19th-century literature, skilled at identifying poetry within unformatted text from Davy’s notebooks. Poetry may exhibit rhythm, rhyme, or archaic language. Focus only on poetic sections with rhythm, rhyme, or archaic language, ignoring scientific or narrative prose. Delimit the poetry by providing the page number (provided in the input), the first few words, and the last few words of the poem segment. Use the following format for your output:

{format_instructions}

Below are examples:

Example 1:
Page Number: 7
Text: "5 Shall I renounce the countenance living in the beauty of expression shall I give up the articulately sounding voice whose gentle sound has so often lull’d me to repose to worship the dry & unmeaning word benevolence no let me live a son a brother & a lover, let me die a father a husband a friend & a father. On breathing the Nitrous oxide On breathing the Nitrous oxide Nitrous oxide Nitrous oxide Not in the ideal dreams of wild desire Have I beheld a rapture wakening form My bosom burns with no unhallowed fire Yet is my cheek with rosy blushes warm. Yet are my eyes with sparkling lustre filled Yet is my murmuring mouth implete with murmuring dying murmuring sound"
Output: {{"page_number": "7", "beginning_words": "Not in the ideal", "ending_words": "murmuring dying murmuring sound"}}

Example 2:
Page Number: 9
Text: "7 The life of the Spinosist The life of the Spinosist The insensate dust is seen to t The insensate dust is seen to t The dust insensate rises into life. The dust insensate rises into life. – The liquid dew is lovely in the flower The liquid dew is lovely in the flower The liquid dew becomes the rosy flower The liquid dew becomes the rosy flower The o The o The Spinosist The Spinosist Lo oër the earth the kindling spirits pour The spark seeds of life that mighty bounteous nature gives. – The liquid dew becomes the rosy flower The sordid dust awakes & moves & lives. – All, All is change, the renovated forms Of ancient things arise & live again. The light of suns the angry breath of storms The everlasting motions of the main Are but the engines of that powerful will. – The eternal link of thoughts, where firm resolves Have ever acted & are acting still. – Whilst age round age & world round world revolves. –"
Output: {{"page_number": "9", "beginning_words": "The dust insensate", "ending_words": "world round world revolves"}}

Example 3:
Page Number: 170
Text: "167 1825 And when the light of life is flying And darkness round us seems to close Nought do we truly know of dying Save sinking in a deep repose And as in sweetest soundest slumber The mind enjoys its happiest dreams And as in stillest night we number Thousands of worlds in starlight beams So may we hope the undying spirit In quitting its decaying form Breaks forth new glory to inherit As lightning from the gloomy storm."
Output: {{"page_number": "170", "beginning_words": "And when the light", "ending_words": "lightning from the gloomy storm"}}

Example 4:
Page Number: 5
Text: [Insert Example 4 non-poetic page text here, e.g., scientific notes from 13C page 5]
Output: {{"page_number": "5", "beginning_words": "", "ending_words": ""}}

Task: Analyze the following text from Davy’s notebook page and provide the structured output using the provided page number.
Page Number: {page_number}
Text: {text}
Output:
"""

prompt = ChatPromptTemplate.from_template(template,partial_variables={"format_instructions": parser.get_format_instructions()})

# Set up the LLM chain
chain = LLMChain(llm=llm, prompt=prompt)


# Function to load and process notebook pages
def process_notebooks(notebook_ids):
    results = []
    base_dir = os.path.join("../..", "preprocessing")
    json_files = ["page_to_text_with_periods.json", "page_to_text.json"]

    for notebook_id in notebook_ids:
        notebook_dir = os.path.join(base_dir, notebook_id)
        for json_file in json_files:
            file_path = os.path.join(notebook_dir, json_file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for page_number, page_text in data.items():
                        response = chain.run(page_number=page_number, text=page_text)
                        parsed_output = parser.parse(response)
                        results.append({
                            "notebook": notebook_id,
                            "page_number": parsed_output.page_number,
                            "beginning_words": parsed_output.beginning_words,
                            "ending_words": parsed_output.ending_words
                        })
    return results


# Main execution
if __name__ == "__main__":
    notebooks_to_process = ['13C', '14E']
    poetry_results = process_notebooks(notebooks_to_process)
    for result in poetry_results:
        print(f"Notebook: {result['notebook']}, Page: {result['page_number']}, "
              f"Beginning: {result['beginning_words']}, Ending: {result['ending_words']}")