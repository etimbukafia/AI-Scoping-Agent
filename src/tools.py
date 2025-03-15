from langchain_core.tools import tool
from utils import analyze_doc
from filehandlers import extract_text_from_doc
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
from typing import Union, Annotated
from server import configs

load_dotenv()

class WriteToMarkdownInput(BaseModel):
    documentation: Union[str, bytes]  # Accept either string or bytes

def write_to_markdown(documentation: Annotated[Union[str, bytes], "The document content to process"]):
    """
    Tool to explain protocol's documentation in markdown.
    Expects a dictionary with a key "documentation" containing the document content (string or bytes).
    """ 
    # If documentation is a dict, extract the document content.
    if isinstance(documentation, dict):
        doc_content = documentation.get("documentation")
    else:
        doc_content = documentation

    client, model = configs.get_client()
    # Process the document content.
    text = extract_text_from_doc(doc_content)
    response = analyze_doc(client, model, text)
    
    # Optionally, write the response to a file.
    with open("files/report.md", "w", encoding="utf-8") as output:
        output.write(response)
    
    return text, response


chat_prompt = """
You are a security expert with deep knowledge of web3 security and smart contract auditing. 
You have access to a vector store containing detailed security reports—including risk classifications, vulnerabilities, and technical assessments. 
When asked a question {user_input}, first review the relevant retrieved report excerpts {docs_content}, then answer the question with concisely and avoid generic responses. 
Provide your answer in a structured and professional manner.
"""

@tool
def chat_with_vectorstore(query: str):
    """
    Tool to retrieve relevant docs relating to the user's query.
    """
    try:
        llm = configs.get_llm()
        vector_store = configs.get_vector_store()
        retrieved_docs = vector_store.similarity_search(query)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        formatted_prompt = chat_prompt.format(user_input=query, docs_content=docs_content)
        response = llm.invoke(formatted_prompt)
        return {"ai_response": response.content}
    except Exception as e:
        logging.exception(f"Error retrieving docs: {str(e)}")
        return {"ai_response": "An error occurred while retrieving documents."}
    
     
    

