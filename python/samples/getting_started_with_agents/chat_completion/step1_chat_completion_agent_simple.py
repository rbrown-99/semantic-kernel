# Copyright (c) Microsoft. All rights reserved.
# For demo purposes only. Not intended for production use.
import asyncio

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

"""
The following sample demonstrates how to create a chat completion agent that
answers user questions using the Azure Chat Completion service. The Chat Completion
Service is passed directly via the ChatCompletionAgent constructor. This sample
demonstrates the basic steps to create an agent and simulate a conversation
with the agent.

The interaction with the agent is via the `get_response` method, which sends a
user input to the agent and receives a response from the agent.
"""

# Simulate a conversation with the agent
USER_INPUTS = [
    "Why is the sky blue?",
    "What is the capital of France?",
]


async def main():
    # 1. Create the agent by specifying the service
    agent = ChatCompletionAgent(
    service=AzureChatCompletion(
        deployment_name="gpt-4o",  
        endpoint="https://azure-openai-rmb.openai.azure.com",  
        api_key="6CePy6190GsjZ7FSKU0fK1LNomKSJbjrCc0oTcYndfSEJvuBU2LbJQQJ99BDACYeBjFXJ3w3AAABACOGTn2s"  
    ),
    name="Assistant",
    instructions="Answer questions about the world in one sentence.",
)

    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        # 2. Invoke the agent for a response
        response = await agent.get_response(
            messages=user_input,
        )
        # 3. Print the response
        print(f"# {response.name}: {response}")

    """
    Sample output:
    # User: Why is the sky blue?
    # Assistant: The sky appears blue because molecules in the Earth's atmosphere scatter shorter wavelengths of 
        sunlight, like blue, more than the longer wavelengths, causing the sky to look blue to our eyes.
    # User: What is the capital of France?
    # Assistant: The capital of France is Paris.
    """


if __name__ == "__main__":
    asyncio.run(main())
