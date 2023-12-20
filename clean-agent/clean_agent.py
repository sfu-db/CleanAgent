import os
import pdb
import pandas as pd
import numpy as np
import streamlit as st
import autogen
import chromadb

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

current_dir = os.getcwd()
files = os.listdir(current_dir)
print("Current Directory:", current_dir)

df = pd.DataFrame({"Name":
                   ["Abby", "Scott", "Scott", "Scott2", np.nan, "NULL"],
                   "AGE":
                   [12, 33, 33, 56,  np.nan, "NULL"],
                   "weight__":
                   [32.5, 47.1, 47.1, 55.2, np.nan, "NULL"],
                   "Admission Date":
                   ["2020-01-01", "2020-01-15", "2020-01-15",
                    "2020-09-01", pd.NaT, "NULL"],
                   "email_address":
                   ["abby@gmail.com","scott@gmail.com", "scott@gmail.com", "test@abc.com", np.nan, "NULL"],
                   "Country of Birth":
                   ["CA","Canada", "Canada", "NULL", np.nan, "NULL"],
                   "Contact (Numbers)":
                   ["1-789-456-0123","1-123-456-7890","1-123-456-7890","1-456-123-7890", np.nan, "NULL" ],

})
df.to_csv('origin_data.csv', index=False)

class CleanAgent(object):
    def __init__(self, df):
        self.CANDIDATE_COLUMN_TYPES = "email, address, phone, date, country"

        self.termination_notice = '\n\nDo not say show appreciation in your responses, say only what is necessary. if "Thank you" or "You\'re welcome" are said in the conversation, then say TERMINATE ' \
                                  'to indicate the conversation is finished and this is your last message.'

        self.COL_ANNOTATOR_SYSTEM_MESSAGE = """You are a helpful AI assistant.
            Please solve the column type annotation task using your language skill.
            Answer the question based on the task and instructions below. 
            If the question cannot be answered using the information provided, answer with 'I donnot know'
            Classify the columns of a given table with only one of the following classes that are seperated with comma: {candidate_column_types}.
                1. Look at the input given to you and make a table out of it.
                2. Look at the cell values in detail.
                3. For each column, select a class that best represents the meaning of all cells in the column.
                4. Answer with the selected class for each columns with the format columnName: class.
            Solve the task step by step if you need to. If a plan is not provided, explain your plan first.
            If the result indicates there is an error, fix the error and output again.
            When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
            Sample rows of the given table is shown as follows: {df}.
            """

        self.PROBLEM = """Use dataprep library to clean the table {path}.\n
            Please follow the three steps:\n
            1. Annotate the type of each column within the five types: {candidate_column_types}. \n
            2. Pick up corresponding clean functions to clean the column.\n
            3. store the cleaned dataframe as csv file named as 'cleaned_data.csv'\n"""

        self.config_list = [{
            'model': 'gpt-4-1106-preview',
            'api_key': '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'
        }]

        self.llm_config = {
            "timeout": 60,
            "cache_seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }

        self.termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

        self.user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            is_termination_msg=self.termination_msg,
            human_input_mode="NEVER",
            system_message="A human admin.",
            code_execution_config=False,
        )

        self.user_proxy_aid = RetrieveUserProxyAgent(
            name="User_Proxy_Assistant",
            is_termination_msg=self.termination_msg,
            system_message="Assistant who has extra content retrieval power for solving difficult problems.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=50,
            retrieve_config={
                "task": "code",
                "docs_path": ["https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_date.ipynb",
                              "https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_email.ipynb",
                              "https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_address.ipynb",
                              "https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_phone.ipynb",
                              "https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_country.ipynb"],
                "chunk_token_size": 1000,
                "model": self.config_list[0]["model"],
                "client": chromadb.PersistentClient(path="/tmp/chromadb"),
                "collection_name": "groupchat",
                "get_or_create": True,
            },
            code_execution_config=False,  # we don't want to execute code in this case.
        )

        self.col_annotator = autogen.AssistantAgent(
            name="Column_Annotator",
            is_termination_msg=self.termination_msg,
            system_message=self.COL_ANNOTATOR_SYSTEM_MESSAGE.format(candidate_column_types=self.CANDIDATE_COLUMN_TYPES, df=df.head()) + self.termination_notice,
            llm_config=self.llm_config,
        )

        self.coder = autogen.AssistantAgent(
            name="Senior_Python_Engineer",
            is_termination_msg=self.termination_msg,
            system_message=f"You are a senior python engineer. Please only output the code.",
            llm_config=self.llm_config,
        )

        self.executor = autogen.UserProxyAgent(
            name="Executor",
            system_message="Executor. Execute the code written by the engineer and report the result.",
            human_input_mode="NEVER",
            code_execution_config={"last_n_messages": 3, "work_dir": "paper"},
        )


    def _reset_agents(self):
        self.user_proxy.reset()
        self.user_proxy_aid.reset()
        self.col_annotator.reset()
        self.coder.reset()
        self.executor.reset()

    def _rag_chat(self, input_path, chat_output):
        self._reset_agents()
        groupchat = autogen.GroupChat(
            agents=[self.user_proxy_aid, self.col_annotator, self.coder, self.executor], messages=[], max_round=12, speaker_selection_method="round_robin"
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

        # Start chatting with boss_aid as this is the user proxy agent.
        self.user_proxy_aid.initiate_chat(
            manager,
            problem=self.PROBLEM.format(candidate_column_types=self.CANDIDATE_COLUMN_TYPES, path=input_path),
            n_results=3,
        )

        for message in groupchat.messages:
            chat_output += f"{message['name']}: {message['content']}\n"
            st.session_state.chat_output = chat_output  # Update session state


    def _norag_chat(self, input_path, chat_output):
        self._reset_agents()
        groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.col_annotator, self.coder, self.executor],
            messages=[],
            max_round=12,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

        # Start chatting with the boss as this is the user proxy agent.
        self.user_proxy.initiate_chat(
            manager,
            message=self.PROBLEM.format(candidate_column_types=self.CANDIDATE_COLUMN_TYPES, path=input_path),
        )

        for message in groupchat.messages:
            chat_output += f"{message['name']}: {message['content']}\n"
            st.session_state.chat_output = chat_output  # Update session state

    def call_rag_chat(self, input_path, annotator_output, code_output, executor_output,  user_input = None):
        self._reset_agents()
        print(user_input)
        if user_input != None:
            self.coder.update_system_message(self.executor.system_message + f"\n Please notice that {user_input}")
        print(self.coder.system_message)
        # print(self.)
        # In this case, we will have multiple user proxy agents and we don't initiate the chat with RAG user proxy agent.
        # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call it from other agents.
        def retrieve_content(message, n_results=3):
            self.user_proxy_aid.n_results = n_results  # Set the number of results to be retrieved.
            # Check if we need to update the context.
            update_context_case1, update_context_case2 = self.user_proxy_aid._check_update_context(message)
            if (update_context_case1 or update_context_case2) and self.user_proxy_aid.update_context:
                self.user_proxy_aid.problem = message if not hasattr(self.user_proxy_aid, "problem") else self.user_proxy_aid.problem
                _, ret_msg = self.user_proxy_aid._generate_retrieve_user_reply(message)
            else:
                ret_msg = self.user_proxy_aid.generate_init_message(message, n_results=n_results)
            return ret_msg if ret_msg else message

        self.user_proxy_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

        llm_config = {
            "functions": [
                {
                    "name": "retrieve_content",
                    "description": "retrieve content for code generation and question answering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                            }
                        },
                        "required": ["message"],
                    },
                },
            ],
            "config_list": self.config_list,
            "timeout": 60,
            "cache_seed": 42,
        }

        for agent in [self.col_annotator, self.coder]:
            # update llm_config for assistant agents.
            # pdb.set_trace()
            agent.llm_config.update(llm_config)

        for agent in [self.user_proxy, self.col_annotator, self.coder, self.executor]:
            # register functions for all agents.
            agent.register_function(
                function_map={
                    "retrieve_content": retrieve_content,
                }
            )

        groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.col_annotator, self.coder, self.executor],
            messages=[],
            max_round=30,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False,
        )

        manager_llm_config = llm_config.copy()
        manager_llm_config.pop("functions")
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config)

        # Start chatting with the boss as this is the user proxy agent.
        # if user_input == None:
        self.user_proxy.initiate_chat(
            manager,
            message=self.PROBLEM.format(candidate_column_types=self.CANDIDATE_COLUMN_TYPES, path=input_path),
        )
        # else:
        #     self.user_proxy.initiate_chat(
        #         manager,
        #         message=self.PROBLEM.format(candidate_column_types=self.CANDIDATE_COLUMN_TYPES, path=input_path) + user_input,
        #     )


        for message in groupchat.messages:
            if message["name"] == "Column_Annotator":
                if len(message['content']) > 0:
                    annotator_output = f"{message['name']}: {message['content']}\n"
                    st.session_state.annotator_output = annotator_output
            elif message['name'] == "Senior_Python_Engineer":
                if len(message['content']) > 0:
                    code_output = f"{message['name']}: {message['content']}\n"
                    st.session_state.code_output = code_output
            elif message['name'] == "Executor":
                executor_output = f"{message['name']}: {message['content']}\n"
                st.session_state.executor_output = executor_output
            # chat_output += f"{message['name']}: {message['content']}\n"
            # st.session_state.chat_output = chat_output  # Update session state


    def run_chat(self, input_path, annotator_output, code_output, executor_output, user_input = None):
        # self._norag_chat(input_path, chat_output)
        # self._rag_chat(input_path, chat_output)
        self.call_rag_chat(input_path=input_path, annotator_output=annotator_output, code_output=code_output, executor_output=executor_output, user_input=user_input)


def save_uploaded_file(uploaded_file):
    # Create a directory to save the file, if it doesn't already exist
    if not os.path.exists('uploaded_files'):
        os.makedirs('uploaded_files')

    # Create a file path
    file_path = os.path.join('uploaded_files', uploaded_file.name)

    # Write the contents of the uploaded file to the new file.
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def main():
    if 'annotator_output' not in st.session_state:
        st.session_state.annotator_output = ""
    if 'code_output' not in st.session_state:
        st.session_state.code_output = ""
    if 'executor_output' not in st.session_state:
        st.session_state.executor_output = ""

    # User uploads a dataset
    with st.sidebar:
        st.title("Data Cleaning System")
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
        show_button = st.button("Show Data")
        clean_button = st.button("Start Cleaning")

    if uploaded_file is not None:

        file_path = save_uploaded_file(uploaded_file)
        st.write(f"File saved at: {file_path}")

        data = pd.read_csv(uploaded_file)

        if show_button:
            st.write(data)
        
        if clean_button:
            clean_agent = CleanAgent(data)
            clean_agent.run_chat(f"{current_dir}/{file_path}", st.session_state.annotator_output, st.session_state.code_output, st.session_state.executor_output)
        
        # col1, col2 = st.columns([0, 4])
        # Display the chat output
        # col1.text_area("Column Annotator Output:", value=st.session_state.annotator_output, height=400)
        # col2.text_area("Python Code:", value=st.session_state.code_output, height=400)
        # col1.subheader("Column Annotator Output:")
        # col1.markdown(st.session_state.annotator_output)
        col2 = st.container()
        col2.subheader("Python Code:")
        col2.code(st.session_state.code_output, language='python')
        # st.text_area("Executor Output:", value=st.session_state.executor_output, height=150)

        # Text dialog for user interaction
        user_input = st.text_input("Enter your query:")
        if user_input:
            # response = user_proxy.run(user_input)
            # st.text_area("Response:", value=response, height=100)
            clean_agent = CleanAgent(data)
            clean_agent.run_chat(f"{current_dir}/{file_path}", st.session_state.annotator_output, st.session_state.code_output, st.session_state.executor_output, user_input=user_input)


        # Further processing based on user feedback
        # ...

if __name__ == "__main__":
    main()
    # clean_agent = CleanAgent(data)
    # clean_agent.run_chat(output_placeholder=st.empty())



