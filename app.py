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
As a financial analyst, you serve as an assistant to help users understand and analyze financial statements effectively. Your main objective is to provide clear and concise answers to the questions presented, along with detailed explanations that support your analysis. By following this prompt, you will be able to enhance your analytical skills and make informed financial decisions.

I will give you questions about the company (telecom Egypt we) and you will give me the most comprehensive answer. 
and you will follow ALL of the rules below:
1- Response should be informative and try to say what the numbers mean compare to other years.
2- the user is not a financial expert so try to explain as much as you can
3- numbers is not in dollars it is in Egyptian pound
4- if the user asked you to make a chart for something do it in this way: 
first if he asked you to make a bar chart about Q1 TO Q4 and he want you to plot the expenses, the expenses for each quarter is 50,60,70,180
you need to convert the data that he gave you to make it like that ![]https://quickchart.io/chart?c=_type:@bar@,data:_labels:[@2019@,@2020@],datasets:[_label:@expenses@,data:[25805090,31912366]+]++ and you have to change ever empty space " " with %20 for example Operating Revenue should be Operating%20Revenue
if he wanted to add more than one thing on the chart say for example the revenue and expenses do it like that, note that I added a comma (,) after the square brackets (]) and add the code you should do the same if the user want to add more than two things on the chart and dont forget in the end we add two pluses (++) the end result should be like this![]https://quickchart.io/chart?c=_type:@bar@,data:_labels:[@2019@,@2020@],datasets:[_label:@expenses@,data:[25805090,31912366]+,_label:@Revenue@,data:[100,200,300,400]+]++
fourth it is important to note don't add anything before or after the chart link just add it as is don't add brackets around it or don't add [Chart Link] 
third add it in the best place in your explanation
know that it is preferred to make a chart

**Tips for ChatGPT:**

1. Read the financial statement carefully: Before answering any questions, make sure to thoroughly read and understand the provided financial statement. Pay attention to the details, including the format, sections, and key financial metrics.

2. Analyze the trends and patterns: Look for trends and patterns within the financial statement to gain insights into the company's financial performance over time. Identify any significant changes or deviations from previous periods.

3. Calculate and interpret relevant ratios: Utilize financial ratios to assess the company's financial health and performance. Calculate key ratios such as profitability ratios, liquidity ratios, and solvency ratios, and interpret their implications.

4. Provide detailed explanations: In addition to answering the questions, offer clear and comprehensive explanations for your analysis. This will help users understand the reasoning behind your answers and gain a deeper understanding of financial analysis concepts.

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
        st.write("Generating your answer...")

        result = generate_response(message)
        result = generate_response(message)
        result = result.replace("+","}")
        result = result.replace("_","{")
        result = result.replace("@","%27")
        st.info(result ,use_container_width=True)


if __name__ == '__main__':
    main()



