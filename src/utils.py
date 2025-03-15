from dotenv import load_dotenv
from pathlib import Path
from typing import Any, List
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
import re
from qdrant_client.http import models
from textwrap import dedent


async def analyze_doc(client, model, input: any):
    folder_dir = Path.cwd() / "prompts"
    with open(folder_dir / "system_prompt.txt", "r", encoding="utf-8") as sp:
        system_prompt = sp.read()
    with open(folder_dir / "human_prompt.txt", "r", encoding="utf-8") as hp:
        human_prompt = hp.read()
        
    # Use the human_prompt to format the input.
    human_prompt_with_input = human_prompt.format(input=input)
    
    chat_response = await client.chat.complete_async(
        model=model,
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": human_prompt_with_input,
            },
        ], stream=False
    ) 

    return chat_response.choices[0].message.content

async def extract_metadata(client, model, doc: Any): 
    system_prompt = """
    You are an expert document classifier and keyword extractor.

    Instructions:
    - Given an input text, extract the most important keywords that will serve as metadata for auditors to find similar projects.
    - Focus only on niche or sector-specific web3 keywords that uniquely describe the project's core offering.
    - Exclude generic crypto terms (e.g., "token", "hack").
    - Also extract the project's name by analyzing the text and include it among the keywords.
    - Return your output as a comma-separated list of keywords.

    Examples:

    Example 1:
    Text:
    "Collar is a completely non-custodial lending protocol that does not rely on liquidations to remain solvent. Collar is powered by solvers instead of liquidators as well as other DeFi primitives like Uniswap v3.
    Disclaimer: This security review does not guarantee against a hack. It is a snapshot in time of Collar Protocol according to the specific commit. Any modifications to the code will require a new security review."
    Keywords:
    "Collar", "lending", "Uniswap v3"

    Example 2:
    Text:
    "Astaria is a NFT Collateralized Lending Market leveraging a novel 3AM Model.
    Disclaimer: This security review does not guarantee against a hack. It is a snapshot in time of astaria-core and astaria-GPL according to the specific commit. Any modifications to the code will require a new security review."
    Keywords:
    "Astaria", "NFT", "Lending", "Market", "3AM Model"

    Example 3:
    Text:
    "Base is a secure and low-cost Ethereum layer-2 solution built to scale the userbase on-chain.
    Solady is an open source project for gas optimized Solidity snippets.
    Disclaimer: This security review does not guarantee against a hack. It is a snapshot in time of Coinbase Solady according to the specific commit. Any modifications to the code will require a new security review."
    Keywords:
    "Base", "Solady", "layer-2"

    Example 4:
    Text:
    "Royco Protocol allows anyone to create a market around any on-chain transaction (or series of transactions). Using Royco, incentive providers may create intents to offer incentives to users to perform the transaction(s) and users may create intents to complete the transaction(s) and/or negotiate for more incentives.
    Disclaimer: This security review does not guarantee against a hack. It is a snapshot in time of Royco according to the specific commit. Any modifications to the code will require a new security review."
    Keywords:
    "Royco", "Market", "Incentive providers"
    """

    chat_response = await client.chat.complete_async(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": doc,
            },
        ], stream=False
    )

    keywords = chat_response.choices[0].message.content.replace('\n', ' ').replace('Keywords:', '').strip()
    metadata = re.findall(r'"([^"]+)"', keywords)
    return metadata

def get_similar_reports(qdrant_client, metadata):
    filter_condition = models.Filter(
    should=[
        models.FieldCondition(
            key="metadata.keywords", 
            match=models.MatchAny(any=metadata)
        )
    ]
    )

    results = qdrant_client.scroll(
        collection_name="security_reports",
        scroll_filter=filter_condition
    )

    similar_reports = get_file_names(results)
    return similar_reports

def flatten_records(obj):
    """
    Recursively yields individual record objects from a nested structure
    (e.g., a tuple containing lists, etc.)
    """
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from flatten_records(item)
    else:
        yield obj

def get_file_names(records: Any) -> List[str]:
    file_names = set()
    for record in flatten_records(records):
        # Check if record has a 'payload' attribute or key.
        if hasattr(record, 'payload'):
            payload = record.payload
        elif isinstance(record, dict) and "payload" in record:
            payload = record["payload"]
        else:
            continue 
        if isinstance(payload, dict) and "file_name" in payload:
            file_names.add(payload["file_name"])
    return list(file_names)


def format_security_report(report_text):
    """
    Format a security report into a clean, structured output
    regardless of input format.
    
    Args:
        report_text (str): The raw text of the security report
    
    Returns:
        str: Formatted report as Markdown
    """

    
    # Remove excessive newlines and whitespace
    cleaned_text = re.sub(r'\n{3,}', '\n\n', report_text.strip())
    
    # Extract title/headline if present
    title_match = re.search(r'^(.+?)(?=\n|$)', cleaned_text)
    title = title_match.group(0) if title_match else "Security Vulnerability Report"
    
    # Extract severity if present
    severity_match = re.search(r'(?i)severity[\s:]*(critical|high|medium|low|informational|info)', cleaned_text)
    severity = severity_match.group(1).capitalize() if severity_match else None
    
    # Extract location/file path if present
    location_match = re.search(r'(?i)(?:location|file|path|contract)[\s:]*([\w\.\/#\-]+\.(?:sol|js|ts|py|jsx|vue)(?:[#:][L\d\-]+)?)', cleaned_text)
    location = location_match.group(1) if location_match else None
    
    # Build formatted output
    formatted_output = f"# {title}\n\n"
    
    if severity or location:
        formatted_output += "## Overview\n\n"
        if severity:
            formatted_output += f"**Severity:** {severity}\n\n"
        if location:
            formatted_output += f"**Location:** `{location}`\n\n"
    
    # Process the main content
    # Remove markdown-like artifacts but keep structure
    content = re.sub(r'(^|\n)#+ ', r'\1### ', cleaned_text)
    
    # Extract description section
    description_match = re.search(r'(?i)## *description\s*(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if description_match:
        description = description_match.group(1).strip()
        formatted_output += f"## Description\n\n{description}\n\n"
    
    # Extract impact section
    impact_match = re.search(r'(?i)## *impact\s*(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if impact_match:
        impact = impact_match.group(1).strip()
        formatted_output += f"## Impact\n\n{impact}\n\n"
    
    # Extract technical details
    technical_match = re.search(r'(?i)## *(technical details|vulnerability details)\s*(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if technical_match:
        technical = technical_match.group(2).strip()
        formatted_output += f"## Technical Details\n\n{technical}\n\n"
    
    # Extract mitigation
    mitigation_match = re.search(r'(?i)## *(mitigation|recommendation|fix|remediation)\s*(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if mitigation_match:
        mitigation = mitigation_match.group(2).strip()
        formatted_output += f"## Mitigation\n\n{mitigation}\n\n"
    
    # If we didn't extract structured sections, include the whole content
    if not any([description_match, impact_match, technical_match, mitigation_match]):
        # Remove the title part we already used
        main_content = re.sub(r'^.+?\n', '', cleaned_text, 1)
        formatted_output += f"## Details\n\n{main_content}\n"
    
    return formatted_output.strip()