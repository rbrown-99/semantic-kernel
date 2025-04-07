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

    @kernel_function(description="Get the current weather in a given city.")
    def get_weather(self, city: Annotated[str, "The city to get weather info for."]) -> Annotated[str, "The current weather conditions."]:
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

            return (
                f"The weather in {city} is {condition} with a temperature of {temp_f}°F "
                f"(feels like {feels_like}°F). Humidity is {humidity}% with winds at {wind_mph} mph."
            )

        except Exception as e:
            return f"Sorry, I couldn't fetch the weather for {city}. Please try again later."


# Simulate a conversation
USER_INPUTS = [
    "Hi there!",
    "I'm going to San Francisco tomorrow. What's the weather like?",
    "How about Tokyo?",
    "Thanks!",
]

async def main():
    service_id = "agent"
    kernel = Kernel()

    # ✅ Register the weather plugin
    kernel.add_plugin(WeatherPlugin(api_key="0096719934824eaa9cc174930250704"), plugin_name="weather")

    # ✅ Connect Azure OpenAI
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name="gpt-4o",  # <- Confirmed from your Azure portal
            endpoint="https://azure-openai-rmb.openai.azure.com",
            api_key="6CePy6190GsjZ7FSKU0fK1LNomKSJbjrCc0oTcYndfSEJvuBU2LbJQQJ99BDACYeBjFXJ3w3AAABACOGTn2s"
        )
    )

    # Enable auto tool calls
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create the agent
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="TravelBot",
        instructions=(
            "You're a travel assistant that helps users plan for their trips. "
            "You can answer questions about the weather using the weather plugin and give packing advice."
        ),
        arguments=KernelArguments(settings=settings),
    )

    # Chat loop
    thread: ChatHistoryAgentThread = None
    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        async for response in agent.invoke(messages=user_input, thread=thread):
            print(f"# {response.name}: {response}")
            thread = response.thread

    await thread.delete() if thread else None

if __name__ == "__main__":
    asyncio.run(main())
