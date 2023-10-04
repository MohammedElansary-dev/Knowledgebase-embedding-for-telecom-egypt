import streamlit as st
import guidance
import urllib.parse

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="we_company_data.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class financial analysist. 
I will give you questions about the company (telecom Egypt we) and you will give me the most comprehensive answer. 
and you will follow ALL of the rules below:
1/ Response should be informative and try to say what the numbers mean compare to other years.
2/ the user is not a financial expert so try to explain as much as you can

Below is what the user asked:
{message}

Here is my answer:
{ai_answer}"""


prompt = PromptTemplate(
    input_variables=["message", "ai_answer"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    ai_answer = retrieve_info(message)
    response = chain.run(message=message, ai_answer=ai_answer)
    chart_output = generate_chart(message)  # Generate chart based on user message
    return response, chart_output




# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Assistant financial analyst for Telecom Egypt", page_icon=":book:")

    st.header("Assistant financial analyst for Telecom Egypt :book:")
    message = st.text_area("user quetion")

    if message:
        st.write("Generating you answer...")
        
        response, chart_output = generate_response(message)

        combined_output = f"{response}\n\n{chart_output}"

        st.write(combined_output, unsafe_allow_html=True)  # unsafe_allow_html is used to render HTML content (chart_output is HTML)



if __name__ == '__main__':
    main()

# 6. Function for generating charts using guidance library
def generate_chart(query):
    def parse_chart_link(chart_details):
        encoded_chart_details = urllib.parse.quote(chart_details, safe='')
        output = "![](https://quickchart.io/chart?c=" + encoded_chart_details + ")"
        return output
    
    examples = [
        {
            'input': "Make a chart of the 5 tallest mountains",
            'output': {"type":"bar","data":{"labels":["Mount Everest","K2","Kangchenjunga","Lhotse","Makalu"], "datasets":[{"label":"Height (m)","data":[8848,8611,8586,8516,8485]}]}}
        },
        {
            'input': "Create a pie chart showing the population of the world by continent",
            'output': {"type":"pie","data":{"labels":["Africa","Asia","Europe","North America","South America","Oceania"], "datasets":[{"label":"Population (millions)","data": [1235.5,4436.6,738.8,571.4,422.5,41.3]}]}}
        }
    ]

    gpt3 = guidance.llms.OpenAI("gpt-3.5-turbo-16k-0613")


    chart_program = guidance(
        '''
        {{#block hidden=True~}}
        You are a world class financial analysist. , You will generate chart output based on natural language;

        {{~#each examples}}
        Q:{{this.input}}
        A:{{this.output}}
        ---
        {{~/each}}
        Q:{{query}}
        A:{{gen 'chart' temperature=0 max_tokens=500}}
        {{/block~}}
        
        Hello, here is the chart you requested:
        {{parse_chart_link chart}}
        ''')

    chart_output = chart_program(query=query, examples=examples, parse_chart_link=parse_chart_link)
    return chart_output
