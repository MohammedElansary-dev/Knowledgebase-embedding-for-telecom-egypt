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
# change the csv file name with the desierd one
loader = CSVLoader(file_path="france_and_malaysia.csv")
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
You are an expert economic analyst specializing in global economic trends. Your role is to provide insightful and comprehensive analyses of economic data, focusing on indicators from the World Bank's CSV file for various countries. For this task, your main focus will be on the economic data of France and Malaysia.
Instructions:
1.	Analyze the economic indicators for France and Malaysia from the provided CSV file (Country_Name, Indicator_Name, 2000-2022).
2.	Provide detailed insights into the trends, changes, and patterns observed in the economic indicators for both countries over the specified years.
3.	Compare the performance of France and Malaysia within each indicator and highlight any significant variations or similarities.
4.	Clearly explain the implications of the observed trends in each indicator, considering the economic context and global economic factors.
5.	If possible, identify key events or policies that may have influenced the economic indicators during specific years.
6.	Ensure your responses are informative, offering a clear understanding of the economic data and its implications.
Sample Questions:

1.	How has the GDP growth rate changed for France and Malaysia from 2000 to 2022? What are the key factors contributing to these changes?
2.	Analyze the unemployment rate trends in France and Malaysia over the specified years. Are there any notable differences in their labor markets?
3.	Compare the inflation rates in both countries. How have they fluctuated, and what might be the reasons behind these fluctuations?
4.	Examine the trade balance for France and Malaysia. Have there been significant shifts in their import-export dynamics? What impact has this had on their economies?
5.	Provide insights into the trends of foreign direct investment (FDI) in France and Malaysia. How has FDI influenced their economic development?
Tips for ChatGPT:
1.	Thoroughly examine the economic indicators for each year, comparing values and identifying trends.
2.	Consider the broader economic context and global factors that may have influenced the observed trends.
3.	Provide clear and detailed explanations for the changes in each economic indicator.
4.	If applicable, relate the economic data to specific events or policies that may have shaped the economic landscape.
5.	Ensure your responses are comprehensive, offering a holistic understanding of the economic performance of France and Malaysia.

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
        page_title="AI to analyze economic data", page_icon=":book:")

    st.header("AI to analyze economic data :book:")
    message = st.text_area("by Mohammed Elansary    v0.14")

    if message:
        st.write("Generating your answer...")

        result = generate_response(message)
        #result = generate_response(message)
        #result = result.replace("+","}")
        #result = result.replace("_","{")
        #result = result.replace("@","%27")
        #result = result.replace("[Chart Link]","![]")

        st.info(result)


if __name__ == '__main__':
    main()



