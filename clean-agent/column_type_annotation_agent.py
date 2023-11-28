from typing import Callable, Dict, Optional, Union

from autogen import AssistantAgent

class ColumnTypeAnnotationAgent(AssistantAgent):
    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
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
    Reply "TERMINATE" in the end when everything is done.
    The given table is shown as follows: {df}.
    """

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        llm_config: Optional[Union[Dict, bool]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        **kwargs,
    ):
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            llm_config=llm_config,
            **kwargs,
        )