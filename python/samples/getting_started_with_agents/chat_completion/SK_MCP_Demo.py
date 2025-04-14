# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import Annotated
import requests

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments, kernel_function

# --- Weather plugin using WeatherAPI.com ---
class WeatherPlugin:
    """Real-time weather information using WeatherAPI.com."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @kernel_function(description="Gets the current weather in a city")
    def get_weather(self, city: Annotated[str, "The city to get weather for"]) -> str:
        url = f"http://api.weatherapi.com/v1/current.json?key={self.api_key}&q={city}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            condition = data["current"]["condition"]["text"]
            temp_f = data["current"]["temp_f"]
            feels_like = data["current"]["feelslike_f"]
            humidity = data["current"]["humidity"]
            wind_mph = data["current"]["wind_mph"]

            result = (
                f"The weather in {city} is {condition} with a temperature of {temp_f}°F "
                f"(feels like {feels_like}°F). Humidity is {humidity}% with winds at {wind_mph} mph."
            )
            return f"[Tool Used: WeatherPlugin]\n{result}"
        except Exception as e:
            print(f"[WeatherPlugin Error] {e}")
            return f"[Tool Used: WeatherPlugin]\nSorry, I couldn't fetch the weather for {city}."

# --- Time plugin using WeatherAPI.com localtime ---
class TimePlugin:
    """Gets local time using WeatherAPI.com."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @kernel_function(description="Gets the local date and time in a city")
    def get_time(self, city: Annotated[str, "The city to get time for"]) -> str:
        url = f"http://api.weatherapi.com/v1/current.json?key={self.api_key}&q={city}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            localtime = data["location"]["localtime"]
            tz_id = data["location"]["tz_id"]

            return f"[Tool Used: TimePlugin]\nThe local time in {city} is {localtime} ({tz_id})."
        except Exception as e:
            print(f"[TimePlugin Error] {e}")
            return f"[Tool Used: TimePlugin]\nSorry, I couldn't fetch the time for {city}."

# --- Simulated conversation ---
USER_INPUTS = [
    "Hi there!",
    "What's the weather and time in San Francisco?",
    "How about Lisbon?",
    "Thanks!",
]

# MCP-style tool definitions (injected into system prompt)
MCP_TOOL_CONTEXT = """
Here are your available tools, described using the Model Context Protocol:

[
  {
    "type": "tool",
    "name": "get_weather",
    "description": "Gets the current weather in a city",
    "parameters": [
      { "name": "city", "type": "string", "description": "The city to get weather for" }
    ]
  },
  {
    "type": "tool",
    "name": "get_time",
    "description": "Gets the local date and time in a city",
    "parameters": [
      { "name": "city", "type": "string", "description": "The city to get time for" }
    ]
  }
]
"""

async def main():
    service_id = "agent"
    kernel = Kernel()

    # ✅ Register plugins
    weather_api_key = "0096719934824eaa9cc174930250704"
    kernel.add_plugin(WeatherPlugin(api_key=weather_api_key), plugin_name="weather")
    kernel.add_plugin(TimePlugin(api_key=weather_api_key), plugin_name="time")

    # ✅ Connect Azure OpenAI
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name="gpt-4o",
            endpoint="https://azure-openai-rmb.openai.azure.com",
            api_key="6CePy6190GsjZ7FSKU0fK1LNomKSJbjrCc0oTcYndfSEJvuBU2LbJQQJ99BDACYeBjFXJ3w3AAABACOGTn2s"
        )
    )

    # ✅ Enable auto tool calling
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # 🧠 Travel Assistant Agent with MCP-enhanced instructions
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="TravelBot",
        instructions=(
            "You're a helpful travel assistant that uses tools to provide real-time information. "
            + MCP_TOOL_CONTEXT +
            "\nWhen responding to users, be sure to mention which tool was used using [Tool Used: PluginName]. "
            "Provide helpful packing advice based on weather and time as well."
        ),
        arguments=KernelArguments(settings=settings),
    )

    # 💬 Chat loop
    thread: ChatHistoryAgentThread = None
    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        async for response in agent.invoke(messages=user_input, thread=thread):
            print(f"# {response.name}: {response}")
            thread = response.thread

    await thread.delete() if thread else None

if __name__ == "__main__":
    asyncio.run(main())
