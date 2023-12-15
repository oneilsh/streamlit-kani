import streamlit as st
import logging
from kani import ChatRole, ChatMessage, Kani
from kani.engines.base import BaseCompletion, Completion
import asyncio
import tempfile
import json
import os
import shutil
from kani import AIParam, ai_function
from typing import Annotated, Optional, List
import pdfplumber



class UIOnlyMessage:
    """
    Represents a that will be displayed in the UI only.

    Attributes:
        data (Any): The data of the message.
        role (ChatRole, optional): The role of the message. Defaults to ChatRole.ASSISTANT.
        icon (str, optional): The icon of the message. Defaults to "📊".
    """

    def __init__(self, func, role=ChatRole.ASSISTANT, icon="📊"):
        self.func = func
        self.role = role
        self.icon = icon


class StreamlitKani(Kani):
    def __init__(self, engine):
        super().__init__(engine)
        self.display_messages = []
        self.conversation_started = False
        self.files = []
        self.tokens_used_prompt = 0
        self.tokens_used_completion = 0

    def render_in_ui(self, data):
        """Render a dataframe in the chat window."""
        self.display_messages.append(UIOnlyMessage(data))

    @ai_function()
    def list_current_files(self):
        """List the files currently uploaded by the user."""
        return self.files
    
    @ai_function()
    def read_text_file(self, file_name: Annotated[str, AIParam(desc="The name of the file to read.")]):
        """Read a text-like file uploaded by the user."""
        for file in self.files:
            if file.name == file_name:
                # if the file is txt-like, use .read()
                if file.type == "text/plain":
                    return file.read()

        return f"Error: file name not found in current uploaded file set."
    
    @ai_function()
    def read_pdf_file(self, 
                      file_name: Annotated[str, AIParam(desc="The name of the file to read.")],
                      pages: Annotated[List[int], AIParam(desc="The pages of the file to read. Optional.")] = None
                      ):
        """Read a PDF file uploaded by the user. Always ask the user if they want to read all pages (default), or a specific set of pages before calling this function."""
        for file in self.files:
            if file.name == file_name:
                # if it's a pdf, use pdfplumber to extract text
                if file.type == "application/pdf":
                    with pdfplumber.open(file) as pdf:
                        if pages == None:
                            return "\n\n".join([page.extract_text() for page in pdf.pages])
                        else:
                            return "\n\n".join([page.extract_text() for page in pdf.pages if page.page_number in pages])
                    
        return f"Error: file name not found in current uploaded file set."
    
    async def get_model_completion(self, include_functions: bool = True, **kwargs) -> BaseCompletion:
        """Overrides the default get_model_completion to track tokens used.
        See https://github.com/zhudotexe/kanpai/blob/cc603705d353e4e9b9aa3cf9fbb12e3a46652c55/kanpai/base_kani.py#L48
        """
        completion = await super().get_model_completion(include_functions, **kwargs)
        self.tokens_used_prompt += completion.prompt_tokens
        self.tokens_used_completion += completion.completion_tokens

        message = completion.message
        # HACK: sometimes openai's function calls are borked; we fix them here
        if (function_call := message.function_call) and function_call.name.startswith("functions."):
            fixed_name = function_call.name.removeprefix("functions.")
            message = message.copy_with(function_call=function_call.copy_with(name=fixed_name))
            return Completion(
                message, prompt_tokens=completion.prompt_tokens, completion_tokens=completion.completion_tokens
            )
        return completion
    
    async def estimate_next_tokens_cost(self):
        """Estimate the cost of the next message (not including the response)"""
        # includes all previous messages, plus the current
        return sum(self.message_token_len(m) for m in await self.get_prompt()) + self.engine.token_reserve + self.engine.function_token_reserve(list(self.functions.values()))



def initialize_app_config(**kwargs):
    _initialize_session_state()
    defaults = {
        "page_title": "Kani AI",
        "page_icon": None,
        "layout": "centered",
        "initial_sidebar_state": "collapsed",
        "menu_items": {
            "Get Help": "https://github.com/monarch-initiative/agent-smith-ai",
            "Report a Bug": "https://github.com/monarch-initiative/agent-smith-ai/issues",
            "About": "Agent Smith (AI) is a framework for developing tool-using AI-based chatbots.",
        }
    }

    st.set_page_config(
        **{**defaults, **kwargs}
    )


def set_app_agents(agents_func, reinit = False):
    if "agents" not in st.session_state or reinit:
        agents = agents_func()
        st.session_state.agents = agents
        st.session_state.agents_func = agents_func

        if not reinit:
            st.session_state.current_agent_name = list(st.session_state.agents.keys())[0]



def serve_app():
    assert "agents" in st.session_state, "No agents have been set. Use set_app_agents() to set agents prior to serve_app()"
    loop = st.session_state.get("event_loop")
    
    loop.run_until_complete(_main())



# Initialize session states
def _initialize_session_state():
    if "logger" not in st.session_state:
        st.session_state.logger = logging.getLogger(__name__)
        st.session_state.logger.handlers = []
        st.session_state.logger.setLevel(logging.INFO)
        st.session_state.logger.addHandler(logging.StreamHandler())

    st.session_state.setdefault("event_loop", asyncio.new_event_loop())
    st.session_state.setdefault("user_api_key", "")
    st.session_state.setdefault("default_api_key", None)  # Store the original API key
    st.session_state.setdefault("show_function_calls", False)
    st.session_state.setdefault("ui_disabled", False)
    st.session_state.setdefault("lock_widgets", False)
    st.session_state.setdefault("upload_chats_processed", False)



# Render chat message
def _render_message(message):
    current_agent_avatar = st.session_state.agents[st.session_state.current_agent_name].get("avatar", None)
    current_user_avatar = st.session_state.agents[st.session_state.current_agent_name].get("user_avatar", None)

    current_action = "*Thinking...*"

    # first, check if this is a UIOnlyMessage,
    # if so, render it and return
    if isinstance(message, UIOnlyMessage):
        if message.role == ChatRole.USER:
            role = "user"
        else:
            role = "assistant"

        with st.chat_message(role, avatar=message.icon):
            message.func()
        return current_action

    if message.role == ChatRole.USER:
        with st.chat_message("user", avatar = current_user_avatar):
            st.write(message.content)

    elif message.role == ChatRole.SYSTEM:
        with st.chat_message("assistant", avatar="ℹ️"):
            st.write(message.content)

    elif message.role == ChatRole.ASSISTANT and message.tool_calls == None:
        with st.chat_message("assistant", avatar=current_agent_avatar):
            st.write(message.content)

    if st.session_state.show_function_calls:
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_arguments = tool_call.function.arguments
                current_action = f"*Checking source ({func_name})...*"
                with st.chat_message("assistant", avatar="🛠️"):
                    st.text(f"{func_name}(params = {func_arguments})")

        elif message.role == ChatRole.FUNCTION:
            current_action = f"*Evaluating result ({message.name})...*"
            with st.chat_message("assistant", avatar="✔️"):
                st.text(message.content)

    
    return current_action

## kani agents have a save method:
    # def save(self, fp: PathLike, **kwargs):
    #     """Save the chat state of this kani to a JSON file. This will overwrite the file if it exists!

    #     :param fp: The path to the file to save.
    #     :param kwargs: Additional arguments to pass to Pydantic's ``model_dump_json``.
    #     """
## so we will return create a JSON file for each agent,
## and loop over all of those, reading them is as JSON so we 
## can eventually return them to the user as a .json file
## we will return a json object keyed by agent name, 
## but only for those that have conversation_started set to True
## finally, we'll create our temp files in an appropirate temp directory,
## removing them when done for security
## 
def _export_chats():
    # create a temp dir
    temp_dir = tempfile.mkdtemp()
    # create a temp file for each agent
    for agent_name, agent in st.session_state.agents.items():
        if agent.conversation_started:
            agent_file = os.path.join(temp_dir, agent_name + ".json")
            agent.save(agent_file)

    # read each agent file as json
    agent_jsons = {}
    for agent_name, agent in st.session_state.agents.items():
        if agent.conversation_started:
            agent_file = os.path.join(temp_dir, agent_name + ".json")
            with open(agent_file) as f:
                agent_jsons[agent_name] = json.load(f)

    # remove the temp dir
    shutil.rmtree(temp_dir)

    return json.dumps(agent_jsons)

## this is the inverse of the above, using kani's 
## correponding load():
    # def load(self, fp: PathLike, **kwargs):
    #     """Load chat state from a JSON file into this kani. This will overwrite any existing chat state!

    #     :param fp: The path to the file containing the chat state.
    #     :param kwargs: Additional arguments to pass to Pydantic's ``model_validate_json``.
    #     """
def _import_chats(parsed_json):
    # reset all agents
    _clear_chat_all_agents()
    # create a temp dir
    temp_dir = tempfile.mkdtemp()
    # create a temp file for each agent
    for agent_name, agent in parsed_json.items():
        agent_file = os.path.join(temp_dir, agent_name + ".json")
        with open(agent_file, "w") as f:
            json.dump(agent, f)

    # read each agent file as json
    for agent_name, agent in parsed_json.items():
        agent_file = os.path.join(temp_dir, agent_name + ".json")
        st.session_state.agents[agent_name]["agent"].load(agent_file)

    # remove the temp dir
    shutil.rmtree(temp_dir)

    st.rerun()

# Handle chat input and responses
async def _handle_chat_input():
    if prompt := st.chat_input(disabled=st.session_state.lock_widgets, on_submit=_lock_ui):

        agent = st.session_state.agents[st.session_state.current_agent_name]['agent']

        user_message = ChatMessage.user(prompt)
        _render_message(user_message)
        agent.display_messages.append(user_message)

        messages = agent.full_round(prompt)

        agent.conversation_started = True

        st.session_state.current_action = "*Thinking...*"

        while True:
            try:
                with st.spinner(st.session_state.current_action):
                    message = await anext(messages)
                    agent.display_messages.append(message)
                    st.session_state.current_action = _render_message(message)
       
                    session_id = st.runtime.scriptrunner.add_script_run_ctx().streamlit_script_run_ctx.session_id
                    info = {"session_id": session_id, "message": message.model_dump(), "agent": st.session_state.current_agent_name}
                    st.session_state.logger.info(info)
            except StopAsyncIteration:
                break

        st.session_state.lock_widgets = False  # Step 5: Unlock the UI
        st.rerun()

def _clear_chat_all_agents():
    set_app_agents(st.session_state.agents_func, reinit = True)

# Lock the UI when user submits input
def _lock_ui():
    st.session_state.lock_widgets = True

# Main Streamlit UI
async def _main():

    with st.sidebar:
        agent_names = list(st.session_state.agents.keys())
        current_agent_name = st.selectbox(label = "**Assistant**", 
                                          options=agent_names, 
                                          key="current_agent_name", 
                                          disabled=st.session_state.lock_widgets, 
                                          label_visibility="visible")
        st.caption(st.session_state.agents[current_agent_name].get("description", ""))
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")

        uploaded_files = st.file_uploader("Upload a document", 
                                          type=["txt", "pdf"], 
                                          accept_multiple_files = True,
                                          )
        
        st.session_state.agents[current_agent_name]["agent"].files = uploaded_files

        st.markdown("---")

        st.button(label = "Clear All Chats", 
                  on_click=_clear_chat_all_agents, 
                  disabled=st.session_state.lock_widgets)
        
        st.checkbox("🛠️ Show calls to external tools", 
                    key="show_function_calls", 
                    disabled=st.session_state.lock_widgets)



    st.header(st.session_state.current_agent_name)

    current_agent_avatar = st.session_state.agents[st.session_state.current_agent_name].get("avatar", None)
    with st.chat_message("assistant", avatar = current_agent_avatar):
        st.write(st.session_state.agents[st.session_state.current_agent_name]['greeting'])

    for message in st.session_state.agents[st.session_state.current_agent_name]["agent"].display_messages:
        _render_message(message)

    await _handle_chat_input()

    if "token_costs" in st.session_state.agents[st.session_state.current_agent_name]:
        prompt_cost = st.session_state.agents[st.session_state.current_agent_name]["token_costs"]["prompt"]
        completion_cost = st.session_state.agents[st.session_state.current_agent_name]["token_costs"]["completion"]
        agent = st.session_state.agents[st.session_state.current_agent_name]['agent']
        cost = (agent.tokens_used_prompt / 1000.0) * prompt_cost + (agent.tokens_used_completion / 1000.0) * completion_cost
        st.caption(f"Chat prompt tokens: {agent.tokens_used_prompt}, completion tokens: {agent.tokens_used_completion}, cost: ${cost:.2f}")
