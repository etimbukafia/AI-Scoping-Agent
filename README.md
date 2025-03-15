# AI-Scoping-Agent

## Overview

Scoping Agent is an AI-powered assistant designed to help security auditors analyze and evaluate smart contract security. It integrates retrieval-augmented generation (RAG) with an LLM to provide answers using both project documentation and past security reports. The agent dynamically determines whether a question should be answered directly from the protocol documentation or if it requires retrieval from a vector database containing security audits.

## Features

Protocol Documentation Context: The agent processes a protocol's documentation upon startup and uses it as context for answering relevant questions.

Retrieval-Augmented Generation (RAG): Queries the Qdrant vector store for security reports when questions require historical vulnerability data.

Smart Routing: Determines whether to use the protocol documentation or security report retrieval based on the nature of the question.

Interactive CLI Chat: Users can ask questions through a CLI interface.

Retrieval Grading: Evaluates the relevance of retrieved documents before generating answers.

## Setup

### Clone the Repository

```
git clone https://github.com/yourusername/scoping-agent.git
cd scoping-agent
```

### Install Dependencies

Using pip:

```pip install -r requirements.txt```

## Usage

1. Set Up Environment Variables
Create a .env file and set up required keys:

```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=<your_qdrant_api_key>
MISTRAL_API_KEY=<your_mistral_api_key>
```

2. Set Up preferred LLM and Embedding Model in the config.py file

3. Run the Agent
Start the agent and provide the path of the protocol documentation file:

```
python app.py /path/to/protocol_document.pdf
```
4. Ask Questions
Once the agent initializes, you can start asking questions in the CLI:
