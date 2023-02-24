from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about courses on toolsqa.com

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """{context} 
Question: {question}

Answer in Markdown:"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["question","context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0.1)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        # condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain
