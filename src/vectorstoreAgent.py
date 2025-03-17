from src.baseAgent import BaseScopingAgent
from typing import Literal
from langchain.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent

class VectorstoreAgent(BaseScopingAgent):
    def __init__(self, *args, task_specific_var="Retrieval",  **kwargs):
        super().__init__(*args, **kwargs)
        self.task_specific_var = task_specific_var
        self.vectorstore_prompt = """
        You are a seasoned security expert specializing in web3 security and smart contract auditing.
        The context is a comprehensive vector store of security reports, which include detailed risk classifications, vulnerabilities, and technical assessments.
        When you receive a question, provide a precise, evidence-based answer that directly references the retrieved context.
        Your answer should be structured clearly (using bullet points, headers, or numbered lists as needed) and avoid generic or vague responses.
        Question: {question}
        Context: {context}
        Answer:
        """

        self.retriever = self.vectorstore.as_retriever()
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_security_reports",
            "Searches the vector store and return information from security reports pertaining to the user's question"
            
        )

        self.tools = [self.retriever_tool]

    def get_system_prompt(self) -> str:
        return f"""
        You are the lead agent in an audit team. 
        You are a security expert with deep knowledge of web3 security and smart contract auditing
        You assist in auditing and identifying vulnerabilities in smart contracts. 
        You are part of a team of other agent that can perform more specialized tasks.
        You are the first in the chain of agents. 

        You do two things:
        - Identify intent
        - Answer user questions if vectorstore tool not needed

        First, you get a user's question or input and identify the intent.

        The intent can be of only two kinds:
        - The user needs information from the vectorstore of past security reports
        - The user needs a direct answer without going to the vectorstore

        Once you understand the user's question and identify it's intent, if the user needs information from the vectorstore, you call the retriever tool to answer the question.


        But if the user doesn't need information from the vectorstore, answer the question yourself.

        Always provide concise, helpful, and evidence-based responses.
        You will use the protocol's documentation as context when you need to answer user's question.
        This doesn't mean you will always use the protocol's documentation to answer the questions, it just means that you always put it into consideration when answering questions. 
        Some questions may not need you to use the documentation as context

        Context: {self.protocol_doc}
        """


    def grade_documents(self, state) -> Literal["generate", "rewrite"] :
        """
        For determining if the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("----CHECKING DOCUMENT RELEVANCY------")

        class grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes or 'no")

        llm_with_tool = self.llm.with_structured_output(grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content #I have to ensure that state.messages[0] is always the user question and state.messages[-1] is the retrieved document.
        context = last_message.content

        scored_result = chain.invoke({"question": question, "context": context})
        score = scored_result.binary_score

        if score == 'yes':
            print("---DECISION: DOCS RELEVANT -----")
            return "generate"
        
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(score)
            return "rewrite"
        

    def rewrite(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        # Grader
        model = self.llm
        response = model.invoke(msg)
        return {"messages": [response]}
    

    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]       
        question = messages[0].content
        last_message = messages[-1]

        context = last_message.content
        
        prompt_template = PromptTemplate(
            template=self.vectorstore_prompt,
            input_variables=["question", "context"]
        )

        # Chain
        rag_chain = prompt_template | self.llm | StrOutputParser()

        # Run   
        response = rag_chain.invoke({"context": context, "question": question})
        return {"messages": [response]}


    def agent(self, state):
        print("---CALL AGENT---")
        system_prompt = self.get_system_prompt()
        print("------SYSTEM PROMPT------")
        print(system_prompt)
        messages = state["messages"]
        # Extracting the content from the first message
        if messages and hasattr(messages[0], "content"):
            user_content = messages[0].content
        else:
            user_content = str(messages[0])
        inputs = {"messages": [("user", user_content)]}
        response = create_react_agent(self.llm, tools=self.tools, prompt=system_prompt).invoke(inputs)
        return response
    
        










