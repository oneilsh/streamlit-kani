# StreamlitKani

A basic [Streamlit](https://streamlit.io/)-based UI for serving one or more [Kani](https://kani.readthedocs.io/en/latest/index.html)-based tool-using LLM agents. 

Features

- Define multiple agents (for different engines and/or functionality)
- Agents can display streamlit objects from custom functions (dataframes, images)

See [demo_app.py](demo_app.py).

## Installation and Use

#### 1. Load libs

Right now this is just a single file, `kani_streamlit.py`. Import from it as

```python
# kani_streamlit imports
import kani_streamlit as ks
from kani_streamlit import StreamlitKani
```

You'll also need:

```python
# for reading API keys from .env file
import os
import dotenv # pip install python-dotenv
dotenv.load_dotenv() 

# kani imports
from typing import Annotated
from kani import AIParam, ai_function
from kani.engines.openai import OpenAIEngine

# streamlit and pandas for extra functionality
import streamlit as st
import pandas as pd
```

#### 2. Page configuration

Here's where we initialize settings for the page. This MUST be called. Parameters here are
passed to `streamlit.set_page_config()`, see more at https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config. 

```python
ks.initialize_app_config(
    page_title = "StreamlitKani Demo",
    page_icon = "🦀", # can also be a URL
    initial_sidebar_state = "expanded", # or "expanded"
    menu_items = {
            "Get Help": "https://github.com/.../issues",
            "Report a Bug": "https://github.com/.../issues",
            "About": "StreamlitKani is a Streamlit-based UI for Kani agents.",
        }
)
```

#### 3. Setup Agents

StreamlitKani agents are Kani agents and work the same, using `@ai_function()` etc. We must
subclass these instead of Kani to work with the Streamlit UI. 

Note the new `self.render_in_ui()`, which the agent can call to render a streamlit component in the chat stream, for example a Pandas dataframe or a video. This should be given a zero-argument function that produces the content, but can refer to data from the outer scope.

Note that the agent does not have access to the content in this chat stream; you can use the return message to describe what has happened for the agent.

```python
class MyKani(StreamlitKani):

    @ai_function()
    def get_weather(
        self,
        location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
    ):
        """Get the current weather in a given location."""

        weather_df = pd.DataFrame({"date": ["2021-01-01", "2021-01-02", "2021-01-03"], 
                                   "temp": [72, 73, 74]})
        self.render_in_ui(lambda: st.write(weather_df))

        mean_temp = weather_df.temp.mean()
        return f"The user has been shown a table of recent dates and temperatures. The mean temperature is {mean_temp}."


    @ai_function()
    def entertain_user(self):
        """Entertain the user by showing a video."""

        self.render_in_ui(lambda: st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))

        return "The video has just been shown to the user, but they have not begun playing it yet. Tell the user you hope it doesn't 'let them down'."
```

Like regular Kani we define an engine:

```python
engine = OpenAIEngine(os.environ["OPENAI_API_KEY"], model="gpt-4-1106-preview")
```

Next we define a *function* that returns a set of agents, as a dictionary keyed by agent name. Also displayed are a short description and greeting; not that these are not part of the agent's conversation stream but shown to the user for information and instruction. The avatar icons can also be URLs to images.

```python
def get_agents():
    return {
        "Demo Agent": {
            "agent": MyKani(engine),
            "greeting": "Hello, I'm a demo assistant. You can ask me the weather, or to play a random video on youtube.",
            "description": "An agent that demonstrates the capabilities of StreamlitKani.",
            "avatar": "🦀", # these can also be URLs
            "user_avatar": "👤",
        },

    }
```

We have to register that agent-creation function with the app:

```python
ks.set_app_agents(get_agents)
```

#### 4. Start app

Finally, we start the app:

```python
ks.serve_app()
```
