from typing import List

from langchain_core.tools import Tool

TOOL_NAME_MAPPING = {
    "Get Issues": "get_issues",
    "Get Issue": "get_issue",
    "Comment on Issue": "comment_on_issue",
    "List open pull requests (PRs)": "list_open_pull_requests",
    "Get Pull Request": "get_pull_request",
    "Overview of files included in PR": "get_pr_files_overview",
    "Create Pull Request": "create_pull_request",
    "List Pull Requests' Files": "list_pr_files",
    "Create File": "create_file",
    "Read File": "read_file",
    "Update File": "update_file",
    "Delete File": "delete_file",
    "Overview of existing files in Main branch": "get_main_branch_files_overview",
    "Overview of files in current working branch": "get_current_branch_files_overview",
    "List branches in this repository": "list_branches",
    "Set active branch": "set_active_branch",
    "Create a new branch": "create_branch",
    "Get files from a directory": "get_directory_files",
    "Search issues and pull requests": "search_issues_and_prs",
    "Search code": "search_code",
    "Create review request": "create_review_request",
    "Get Latest Release": "get_latest_release",
    "Get Releases": "get_releases",
    "Get Release": "get_release",
}

def rename_tool(tool):
    if tool.name in TOOL_NAME_MAPPING:
        tool.name = TOOL_NAME_MAPPING[tool.name]
    return tool

def get_issue_tools(tools: List[Tool]) -> List[Tool]:
    issue_tool_names = [
        "get_issues",
        "comment_on_issue",
        "search_issues_and_prs",
        "read_file",
    ]
    return [tool for tool in tools if tool.name in issue_tool_names]

def get_code_review_tools(tools: List[Tool]) -> List[Tool]:
    code_review_tool_names = [
        "list_open_pull_requests",
        "get_pull_request",
        "list_pr_files",
        "read_file",
        "search_issues_and_prs",
        "create_pull_request",
        "get_pr_files_overview",
        "search_code"
    ]
    return [tool for tool in tools if tool.name in code_review_tool_names]

def get_release_tools(tools: List[Tool]) -> List[Tool]:
    release_tool_names = [
        "get_latest_release",
        "list_open_pull_requests",
        "get_pr_files_overview",
        "read_file",
    ]
    return [tool for tool in tools if tool.name in release_tool_names]


def get_documentation_tools(tools: List[Tool]) -> List[Tool]:
    documentation_tool_names = [
        "get_main_branch_files_overview",
        "get_current_branch_files_overview",
        "create_branch",
        "get_directory_files",
        "read_file",
        "update_file",
        "create_file",
        "search_code",
        "set_active_branch",
        "create_pull_request",
    ]
    return [tool for tool in tools if tool.name in documentation_tool_names]
