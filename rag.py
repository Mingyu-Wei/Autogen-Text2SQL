
import os
import pandas as pd
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb

assistant = AssistantAgent(
    name="Text2SQL Assistant",
    system_message="You are a helpful Text2SQL assistant capable of generating SQL queries from given database schemas and questions.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": "OPENAI_API_KEY",
            }
        ],
    },
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": ["./schema.txt"],  # Preprocessed schema file
        "custom_text_types": ["txt"],  # Specify text format
        "chunk_token_size": 2000,
        "model": "all-mpnet-base-v2",  # Model for embedding generation
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "embedding_model": "all-mpnet-base-v2",
        "get_or_create": True,
    },
    code_execution_config=False,
)

# Load schema data into RAGProxyAgent
# schema_text = preprocess_schema("schema.xlsx")

# Define a user question
natural_language_question = '''
Inclusion
1.	Interrupted IVC (Surg hx: interrupted IVC, YES)
2.	Azygous continuation who underwent hepato-azygous shunting (surg hx: hepatic vein to azygous baffle)
3.	SVC and Azygous Flow available (CMR Form: SVC flow and fontan flow)
4.	No Baffle Leak (CMR form: baffle leak not checked)
5.	Collateral Flow < 40% (this will be too hard for u to find)
6.	No Thrombosis (hx form: no thrombus)

Exclusion
1.	Extracardiac
2.	Lateral Tunnel
3.	Thrombosis
'''

# Start a chat with retrieval augmentation
response = ragproxyagent.initiate_chat(
    assistant,
    message=ragproxyagent.message_generator,
    problem=natural_language_question,
)
# Extract the SQL content from the response

