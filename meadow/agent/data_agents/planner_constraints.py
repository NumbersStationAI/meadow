"""Constraints for data agents that are good to add to planner."""

from meadow.agent.agent import SubTask


def attribute_detector_constraints(parsed_plan: list[SubTask], user_input: str) -> str:
    """Constraints for AttributeDetector agent."""
    error_message = None
    for pl in parsed_plan:
        if pl.agent.name == "AttributeDetector" and "{" in pl.prompt:
            error_message = f"AttributeDetector agent cannot have any replacement tags in the instruction. instruction={pl.prompt}."
    return error_message


def sql_agent_constraints(parsed_plan: list[SubTask], user_input: str) -> str:
    """Constraints for SQLGenerator and SQLPlanner agents."""
    error_message = None
    if sum([p.agent.name == "SQLGenerator" for p in parsed_plan]) > 1:
        error_message = "SQLGenerator agent can only be used once. Please rewrite to just have a single <agent>SQLGenerator</agent> in the plan."
    if len(parsed_plan) == 1 and parsed_plan[0].agent.name == "SQLGenerator":
        if parsed_plan[0].prompt != user_input:
            error_message = f"SQLGenerator agent must have the exact user input as its instruction. Please make the instruction to SQLGenerator be <instruction>{user_input}</instruction>."
    for p in parsed_plan:
        if (
            p.agent.name == "SQLGenerator"
            and "SELECT" in p.prompt
            and "FROM" in p.prompt
        ):
            error_message = "Please avoid giving SQL to the SQLGenerator. It requires an input instructon for what question to answer."
    return error_message


def question_refiner_constraints(parsed_plan: list[SubTask], user_input: str) -> str:
    """Constraints for QuestionRefiner agent."""
    error_message = None
    for i, pl in enumerate(parsed_plan):
        if pl.agent.name == "QuestionRefiner":
            if i < len(parsed_plan) - 1:
                next_pl = parsed_plan[i + 1]
                if next_pl.agent.name != "SQLGenerator":
                    error_message = (
                        "QuestionRefiner agent must be followed by SQLGenerator agent."
                    )
                if "{" not in next_pl.prompt:
                    error_message = "SQLGenerator agent must have the output of the QuestionRefiner as its instruction. Please use the {stepXX} tag in the SQLGenerator instruction."
            else:
                error_message = (
                    "QuestionRefiner agent must be followed by SQLGenerator agent."
                )
    return error_message


def metadata_question_constraints(parsed_plan: list[SubTask], user_input: str) -> str:
    """Constraints for MetadataQuestion agent."""
    error_message = None
    for pl in parsed_plan:
        if pl.agent.name == "MetadataQuestion" and len(parsed_plan) > 1:
            error_message = "MetadataQuestion agent can only be used as a solo agent. Please do not add any other agents."
    return error_message


def column_generator_constraints(parsed_plan: list[SubTask], user_input: str) -> str:
    """Constraints for MetadataQuestion agent."""
    error_message = None
    for i, pl in enumerate(parsed_plan):
        if pl.agent.name == "ColumnGenerator" and not any(
            pl2.agent.name == "SingleTableSelectorAgent" for pl2 in parsed_plan[:i]
        ):
            error_message = (
                "ColumnGenerator agent must be preceded by SingleTableSelectorAgent."
            )
    return error_message
