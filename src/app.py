import sys
import asyncio
import logging
from langchain_core.messages import HumanMessage

# Import the new CompleteAgent class
from newAgent import ScopingAgent
# Import server module for configs
import server

logging.basicConfig(level=logging.INFO)

async def main():
    # Expect the protocol document path as the first command line argument.
    if len(sys.argv) < 2:
        print("Usage: python main.py <protocol_document_path>")
        sys.exit(1)
    
    protocol_document_path = sys.argv[1]

    # Create and initialize the agent with the server configs
    agent = await ScopingAgent.create(config_module=server)
    print("\n--- Agent Initialized---")
    
    # Process the protocol document
    protocol_text = None
    try:
        with open(protocol_document_path, "rb") as doc_file:
            similar_reports, protocol_text, markdown_response = await agent.process_protocol(doc_file)
    except Exception as e:
        logging.exception("Error processing protocol document: %s", e)
        print("Failed to process protocol document.")
        sys.exit(1)
    
    # Display the generated markdown explanation.
    print("\n--- Protocol Explanation Completed---")
    print("-----------------------------\n")
    print("\n--- Similar Reports: ---")
    print(", ".join(similar_reports))
    print("-----------------------------\n")

    agent.projectDoc = protocol_text
    
    # Now, start an interactive chat loop for questions.
    print("Enter your questions about the protocol (type 'exit' to quit):")
    while True:
        user_query = input("Your question: ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            continue
        
        try:
            # Create initial state with human message
            initial_state = {
                "messages": [
                    HumanMessage(content=user_query)
                ],
                "next": "direct"
            }
            
            # Get response using the agent's method
            result = agent.response_from_ai(initial_state)
            
            print("\n=== Agent Response ===")
            print(result["ai_response"])
            print("======================")
        except Exception as e:
            logging.exception("Error processing query: %s", e)
            print("An error occurred while processing your query.")
    
    print("Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())

