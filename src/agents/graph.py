from enum import Enum
from typing import Annotated

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, InjectedToolCallId
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from .utils import (
    rename_tool,
    get_issue_tools,
    get_release_tools,
    get_code_review_tools,
    get_documentation_tools,
)

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini")
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github, include_release_tools=True)
tools = [rename_tool(t) for t in toolkit.get_tools()]

# Creating Subagents
_documentation_agent = create_agent(
    model=llm,
    tools=get_documentation_tools(tools),
    name="documentation_agent",
    system_prompt=(
        "You are a documentation agent. You do not draft documentation in chat — you commit it to the "
        "repository and open a pull request for it.\n\n"
        "Follow this procedure for every documentation task:\n"
        "1. Inspect the repo with get_main_branch_files_overview / get_directory_files / read_file so you "
        "know whether the target file already exists and what it currently says.\n"
        "2. Call create_branch with a short descriptive name (e.g. 'docs/update-readme'). If that branch "
        "already exists, call set_active_branch on it instead. Never work directly on the main branch — "
        "create_pull_request fails when the active branch is the base branch.\n"
        "3. Write the content with create_file (new file) or update_file (existing file). This step is "
        "mandatory. Returning documentation text without writing it to the branch is a failed task.\n"
        "4. Call create_pull_request to open the PR from your branch. Pass the title on the first line and "
        "the description after a blank line.\n\n"
        "Finish by reporting the branch name, the file path you wrote, and the PR URL. If a tool call "
        "fails, report the exact error rather than continuing to the next step."
    ),
)

_release_notes_agent = create_agent(
    model=llm,
    tools=get_release_tools(tools),
    name="release_notes_agent",
    system_prompt=(
        "You are a release notes agent. Your task is to generate comprehensive and clear release notes for "
        "new software releases. Use the provided tools to gather information about the latest releases, "
        "including pull requests, commits, and issues. Ensure that the release notes highlight key features, "
        "bug fixes, and any other important changes in a user-friendly manner. "
        "Always include the full release notes text in your final message so the coordinator can see it."
    ),
)

_issue_agent = create_agent(
    model=llm,
    tools=get_issue_tools(tools),
    name="issue_agent",
    system_prompt=(
        "You are an issue management agent. Your task is to help manage and resolve issues within the project. "
        "Use the provided tools to search for existing issues, retrieve detailed information about specific "
        "issues, and add comments or updates as necessary. Ensure that issues are addressed promptly and that "
        "all relevant information is documented clearly. "
        "Always include a summary of all actions taken in your final message so the coordinator can see it."
    ),
)

_code_review_agent = create_agent(
    model=llm,
    tools=get_code_review_tools(tools),
    name="code_review_agent",
    system_prompt=(
        "You are a code review agent. Your task is to assist in reviewing code changes and pull requests "
        "within the project. Use the provided tools to list open pull requests, examine the files changed in "
        "each pull request, and read specific files as needed. Provide constructive feedback on code quality, "
        "adherence to coding standards, and potential improvements to ensure high-quality contributions. "
        "Always include your full review in your final message so the coordinator can see it."
    ),
)

# Enum for type-safe dispatch
class AgentName(str, Enum):
    DOCUMENTATION = "documentation_agent"
    RELEASE_NOTES = "release_notes_agent"
    ISSUE = "issue_agent"
    CODE_REVIEW = "code_review_agent"


# Subagent-as-tool wrappers
@tool(
    "documentation_agent",
    description=(
        "Handles project documentation end to end: writes/updates README files, API docs, and other "
        "project documentation, commits them to a new branch, and opens the pull request for them. "
        "Pass a clear description of the documentation task to perform."
    ),
)
def call_documentation_agent(
    task: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = _documentation_agent.invoke({"messages": [{"role": "user", "content": task}]})
    return Command(update={
        "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)]
    })


@tool(
    "release_notes_agent",
    description=(
        "Generates comprehensive release notes for new software versions by inspecting the latest "
        "releases, pull requests, and issues. Pass the release scope or version as the task."
    ),
)
def call_release_notes_agent(
    task: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = _release_notes_agent.invoke({"messages": [{"role": "user", "content": task}]})
    return Command(update={
        "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)]
    })


@tool(
    "issue_agent",
    description=(
        "Manages and resolves project issues: searching issues, retrieving issue details, and adding "
        "comments. Pass a clear description of the issue management task to perform."
    ),
)
def call_issue_agent(
    task: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = _issue_agent.invoke({"messages": [{"role": "user", "content": task}]})
    return Command(update={
        "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)]
    })


@tool(
    "code_review_agent",
    description=(
        "Reviews code changes and pull requests: lists open PRs, examines changed files, reads "
        "source files, and provides constructive feedback. Pass a description of the review task."
    ),
)
def call_code_review_agent(
    task: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = _code_review_agent.invoke({"messages": [{"role": "user", "content": task}]})
    return Command(update={
        "messages": [ToolMessage(content=result["messages"][-1].content, tool_call_id=tool_call_id)]
    })


# Main coordinator agent
github_agent = create_agent(
    model=llm,
    tools=[
        call_documentation_agent,
        call_release_notes_agent,
        call_issue_agent,
        call_code_review_agent,
    ],
    system_prompt=(
        "You are a GitHub project coordinator that delegates tasks to specialized subagents via tools. "
        "Available subagents:\n"
        "- documentation_agent: writes and maintains README files, API docs, and other project "
        "documentation, commits them to a branch, and opens the PR for them\n"
        "- release_notes_agent: generates release notes for new software versions\n"
        "- issue_agent: searches, retrieves, and comments on project issues\n"
        "- code_review_agent: reviews pull requests and provides code quality feedback\n\n"
        "Delegate each task to the most appropriate subagent. A documentation change and the PR that "
        "ships it are a single task for documentation_agent — do not split them across subagents, and do "
        "not ask code_review_agent to open a PR for work another subagent produced. "
        "Invoke one subagent at a time. Do not perform any GitHub operations yourself — "
        "always use the provided tools to delegate work."
    ),
)


if __name__ == "__main__":
    res = github_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Create a documentation and raise a PR for it.",
            }
        ]
    })
    print(res)
