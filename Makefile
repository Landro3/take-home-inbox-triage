.PHONY: serve audit

# Start the mock client API on http://127.0.0.1:8099
serve:
	uv run python -m uvicorn mock_api.server:app --host 127.0.0.1 --port 8099 --reload

# Peek at the side effects your skill has produced against a running server
audit:
	@curl -s http://127.0.0.1:8099/_audit | uv run python -m json.tool
