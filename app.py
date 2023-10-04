import streamlit as st
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
now You are a world class financial analysist. 
I will give you questions about the company (telecom Egypt we) and you will give me the most comprehensive answer. 
and you will follow ALL of the rules below:
1/ Response should be informative and try to say what the numbers mean compare to other years.
2/ the user is not a financial expert so try to explain as much as you can
3/ numbers is not in dollars it is in Egyptian pound
4/ if the user asked you to make a chart for somthing do it in this way: 
first if he asked you to make a bar chart of the Q1, Q2, Q3, and Q4 for the year 2022 and he want you to plot the expenses and revenue and the expenses for each quarter is 50,60,70,180 and for the revenue is 100,200,300,400, you need to convert the data that he gave you to make it like that {type:'bar',data:{labels:['Q1','Q2','Q3','Q4'], datasets:[{label:' expenses ',data:[50,60,70,180]},{label:'Revenue',data:[100,200,300,400]}]}}
second add this url to the start of it https://quickchart.io/chart?c= so that it dose look like that in the end https://quickchart.io/chart?c={type:'bar',data:{labels:['Q1','Q2','Q3','Q4'], datasets:[{label:' expenses ',data:[50,60,70,180]},{label:'Revenue',data:[100,200,300,400]}]}}
third add it in the best place in your explanation
fourth it is important to note don't add any thing before or after the chart link just add it as is don't add brackets around it or don't tell me that this is a chart link like this [Chart Link] just but the link  
know that it is preferred to make a chart 

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
    return response



# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Assistant financial analyst for Telecom Egypt", page_icon=":book:")

    st.header("Assistant financial analyst for Telecom Egypt :book:")
    message = st.text_area("by Mohammed Elansary")

    if message:
        st.write("Generating you answer...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
