import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():

    # LLM
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o"
    )

    # Agent 1
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        system_message="""
        You are a technical researcher.
        Research the given topic.
        """
    )

    # Agent 2
    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message="""
        You are a technical writer.
        Convert the research into a simple explanation.
        When finished, say TERMINATE.
        """
    )

    # Stop condition
    termination = TextMentionTermination(
        "TERMINATE"
    )

    # Team
    team = RoundRobinGroupChat(
        [researcher, writer],
        termination_condition=termination
    )

    # Start
    result = await team.run(
        task="Explain RAG to a beginner."
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())