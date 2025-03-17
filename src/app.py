import sys
import asyncio
import logging
from langchain_core.messages import HumanMessage

# Import the new CompleteAgent (or equivalent) class creation function
from src.main import create_agent

logging.basicConfig(level=logging.INFO)

async def main():
    # Expect the protocol document path as the first command line argument.
    if len(sys.argv) < 2:
        print("Usage: python main.py <protocol_document_path>")
        sys.exit(1)
    
    protocol_document_path = sys.argv[1]

    # Create and initialize the agent with the server configs.
    # The agent's initialization now processes the protocol document and similar reports.
    agent = await create_agent(protocol_document_path)
    print("\n--- Agent Initialized---")

    # Access the outputs that were processed during initialization.
    # For example, assume the agent now has attributes `protocol_doc` and `similar_reports`
    print("\n--- Protocol Explanation Completed---")
    print("-----------------------------\n")
    print("\n--- Similar Reports: ---")
    # Ensure similar_reports is a list of strings
    print(", ".join(agent.similar_reports))
    print("-----------------------------\n")

    # Now, start an interactive chat loop for questions.
    print("Enter your questions about the protocol (type 'exit' to quit):")
    while True:
        user_query = input("Your question: ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            continue

        user_input = {"messages": [("user", user_query)]}
        
        try:
            # Get response using the agent's method
            result = agent.workflow_graph.invoke(user_input)
            
            print("\n=== Agent Response ===")
            print(result)
            print("======================")
        except Exception as e:
            logging.exception("Error processing query: %s", e)
            print("An error occurred while processing your query.")
    
    print("Goodbye!")
    
    # When finished, ensure to clean up resources.
    agent.close()

if __name__ == "__main__":
    asyncio.run(main())