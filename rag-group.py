
import os
from typing_extensions import Annotated
import pandas as pd
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from executor import executor, code_executor_agent
import chromadb

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


llm_config={
    "timeout": 600,
    "cache_seed": 42,
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": ""
        }
    ],
}

user = UserProxyAgent(
    name="User",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    code_execution_config={"executor": executor},
    default_auto_reply="Please read the code provided by Coder, the reviews provided by the reviewer, the execution result provided by User. If you want to execute code, provide a code block; Only reply `TERMINATE` when the task is done.",
    description="The user who asks questions and runs code snippets.",
)

ragproxyagent = RetrieveUserProxyAgent(
    name="Assistant",
    human_input_mode="NEVER",
    is_termination_msg=termination_msg,
    max_consecutive_auto_reply=3,
    default_auto_reply="If you want more information, reply `UPDATE CONTEXT`; If you want to execute code, provide a code block; Only reply `TERMINATE` when the task is done.",
    retrieve_config={
        "task": "code",
        "docs_path": ["./schema_pt_master.txt", "./schema_pt_hx.txt", "./schema_cmr.txt", "./schema_status.txt", "./schema_echo.txt", "./schema_est.txt", "./schema_cath.txt", "./schema_cct.txt"],  # Preprocessed schema file
        "custom_text_types": ["txt"],  # Specify text format
        "chunk_token_size": 2000,
        "model": "text-embedding-3-small",  # Model for embedding generation
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "embedding_model": "text-embedding-3-small",
        "get_or_create": True,
    },
    code_execution_config={"executor": executor},
    description="Assistant who has extra content retrieval power for solving difficult problems. You know the schema of the database.",
)

pm = autogen.AssistantAgent(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message="You are a product manager. You design and plan the code project for the coder. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
    description="Product Manager who can design and plan the code project.",
)

coder = AssistantAgent(
    name="Coder",
    is_termination_msg=termination_msg,
    system_message="""You are the Coder, a retrieve augmented python Coder who is good at writing SQL queries and visualization, you provide python code to answer questions. There should be 8 tables in the database.
You are given database schemas and questions. Solve tasks using your coding and language skills. Reply `TERMINATE` in the end when everything is done.
You answer user's questions based on your own knowledge and the context provided by the user. Use SQLite3 to interact with the database and after you get the result, please visualize them (output markdown tables or plot charts. You choose what fits the best). Use matplotlib for visualization but don't use `plot.show()`, please save it as a local image file called `output-i.png` instead, where i is the index of images, starting from 1 (in case you need to plot multiple images. But in most cases don't plot unnecessary images).
After fetching the data and visutalizing it, you also need to output the data as a table in markdown format. You can do it also in the Python code.
If you give out information retrieving request, the user will reply you with retrieved information. You can use the retrieved information to generate code.
If you give out a Python code snippet, the user will execute the code and reply you with the execution result. You can use the execution result or errors to improve your code until you feel it's good enough. Review your own code with the execution results.
In the following cases, suggest python code (the code part has to be in a python coding block) for the user to execute. Finish the task step by step.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Please only write Python code. The question is about the data in the medical database. The database is stored in a file named 'medical.db'. The database schema is provided in the schema.txt file. The user can execute the code in the Python environment. Please use the 'sqlite3' library to interact with the database.
You can generate the code in this format:
```python
import sqlite3

conn = sqlite3.connect('./medical.db')
c = conn.cursor()
res = c.execute('{HERE IS YOUR SQL QUERY}')
print(res)
conn.close()
```""",
    llm_config=llm_config,
    description="Python Coder who can write code to solve problems and answer questions.",
)

reviewer = AssistantAgent(
    name="Reviewer",
    is_termination_msg=termination_msg,
    system_message="""Critic. You are a helpful assistant highly skilled in evaluating the quality of SQLite3 queries and visualization code by providing a score from 0 (bad) - 10 (good) while providing clear rationale. TOU MUST CONSIDER BEST SQLITE3 PRACTICES. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each visualization (if there is any visualization task). Specifically, you can carefully evaluate the code across the following dimensions
- bugs (bugs):  are there bugs, logic errors, syntax error or typos? Are there any reasons why the code may fail to compile? How should it be fixed? If ANY bug exists, the bug score MUST be less than 5.
- SQL queries (sql): Is the SQLite3 codes valid? Are they appropriate? Is it bug-free? Does it address the question proposed by the user? E.g., is the table name correct? Are the columns correct? Are the conditions correct? Are the joins correct? Are the selected fields correct?
- Goal compliance (compliance): how well the code meets the specified goals? e.g. does the code answer the user's question? Does the code provide the correct information?
- Visualization type (type): CONSIDERING BEST PRACTICES, is the visualization type appropriate for the data and intent? Is there a visualization type that would be more effective in conveying insights? If a different visualization type is more appropriate, the score MUST BE LESS THAN 5. If there is no visualization task, the score should be 10. Visualization includes but is not limited to: markdown tables, human-readable data format, bar charts, line charts, scatter plots, pie charts, histograms, box plots, heatmaps, etc.
- aesthetics (aesthetics): Are the aesthetics of the visualization appropriate for the visualization type and the data? Is the legend well placed? Are the labels clear? Is the color scheme appropriate? Is the font size appropriate? Is the title clear? Is the axis label clear? Is the visualization easy to read and understand?

YOU MUST PROVIDE A SCORE for each of the above dimensions.
{bugs: 0, transformation: 0, compliance: 0, type: 0, encoding: 0, aesthetics: 0}
Do not suggest code.
Then, based on the critique above, suggest a concrete list of actions that the coder should take to improve the code.
Finnaly, if the code execute successfully and output a table of data, you need to summarize the data in a few sentences. E.g. conclude the data, provide insights, or answer the user's question.
If the code doesn't execute successfully or output a table of correct data, you need to give advice and suggestions about how to modify the code.
Don't tell user to do any action other than executing the code you provide. You have an RAG assistant who can provide you with the database schema information.
Think before you answer. Double check your answers. Do it step by step.
""",
    llm_config=llm_config,
    # code_execution_config={"executor": executor},
    description="Python Code Reviewer who can review the code.",
)

# Define a user question
# natural_language_question = '''
# Can you help me pull a patient list? 

# BCH FORCE pts
# Contrast on CMR
# LGE present on scan (most interested in Subendocardial fibroelastosis AND LGE in the bigger systemic ventricle)
# Will also need you to pull: cardiac dx, fontan type at CMR, location and description of LGE, any arrhythmia history and pt status (alive, transplant, dead)
# '''
natural_language_question = "Give me the patients' names, with their Peak RER value which should be a float. Also give me a bar chart according to their Peak RER value."

def _reset_agents():
    user.reset()
    ragproxyagent.reset()
    coder.reset()
    reviewer.reset()

def rag_chat():
    # DISABLED
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[ragproxyagent, coder, reviewer], messages=[], max_round=12, speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    ragproxyagent.initiate_chat(
        manager,
        message=ragproxyagent.message_generator,
        problem=natural_language_question,
        n_results=8,
    )

def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 8,
    ) -> str:
        ragproxyagent.n_results = n_results  # Set the number of results to be retrieved.
        _context = {"problem": message, "n_results": n_results}
        ret_msg = ragproxyagent.message_generator(ragproxyagent, None, _context)
        return ret_msg or message

    ragproxyagent.human_input_mode = "NEVER"  # Disable human input for ragproxyagent since it only retrieves content.

    for caller in [coder, reviewer]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve database schema for code generation and question answering. PLEASE specify the content you are looking for so the RAG can return it, such as names, blood pressure etc.", api_style="function"
        )(retrieve_content)

    for executor in [user, coder]:
        executor.register_for_execution()(d_retrieve_content)

    groupchat = autogen.GroupChat(
        agents=[user, coder, reviewer],
        messages=[],
        max_round=15,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    user.initiate_chat(
        manager,
        message=natural_language_question,
    )

call_rag_chat()