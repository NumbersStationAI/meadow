from meadow.client.schema import FunctionArgSpec, ToolSpec


def test_tool_spec_serialization() -> None:
    """Test serialization of ToolSpec."""

    tool_spec = ToolSpec(
        name="test_tool",
        description="Test tool",
        function_args=[
            FunctionArgSpec(
                name="arg1",
                description="First argument",
                type="string",
                required=True,
            ),
            FunctionArgSpec(
                name="arg2",
                description="Second argument",
                type="number",
                required=False,
            ),
            FunctionArgSpec(
                name="arg3",
                description="Third argument",
                type="array[string]",
                required=False,
            ),
        ],
    )

    expected = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {
                        "description": "First argument",
                        "type": "string",
                    },
                    "arg2": {
                        "description": "Second argument",
                        "type": "number",
                    },
                    "arg3": {
                        "description": "Third argument",
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["arg1"],
            },
        },
    }

    assert tool_spec.model_dump() == expected
