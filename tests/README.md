# Test Suite

This directory contains the test suite for the Envision Agent project.

## Test Files

- **`test_api.py`** - Tests for the `CustomEnvisionAgent` class and core agent functionality
- **`test_lambda_function.py`** - Tests for the Lambda proxy function (`agentcore_proxy.py`)

## Running Tests

### Quick Test Run
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=custom_agent --cov-report=term-missing
```

### Using Test Script
```bash
# Run all tests
./run_tests.sh all

# Run with coverage report
./run_tests.sh coverage

# Run full suite (tests + linting + coverage)
./run_tests.sh full
```

## Test Coverage

The test suite focuses on the core agent functionality (`custom_agent.py`) and excludes:
- CLI scripts (`agent_cli.py`, `runtime_agent_main.py`)
- Test files themselves
- Lambda functions (tested separately in `test_lambda_function.py`)
- Infrastructure files
- Utility scripts

Target coverage: 60% of core agent code.

## Test Categories

- **Unit tests**: Test individual functions and methods with mocks
- **Integration tests**: Test component interactions with AgentCore memory
- **Async tests**: Test asynchronous agent operations (queries, memory)

## Key Test Features

### CustomEnvisionAgent Tests (`test_api.py`)
- ✅ Agent initialization and configuration
- ✅ AWS client creation (Bedrock runtime, agent runtime)
- ✅ System prompt validation
- ✅ Async query processing with mocked responses
- ✅ Memory operations (get/update)
- ✅ Knowledge base retrieval with RAG
- ✅ Text extraction from various response formats

### Lambda Function Tests (`test_lambda_function.py`)
- ✅ CORS preflight handling
- ✅ Request validation and error handling
- ✅ AgentCore service integration
- ✅ Response processing (event-stream, JSON)
- ✅ Error response formatting

## Dependencies

Tests require:
- `pytest`
- `pytest-cov` (for coverage)
- `pytest-mock` (for mocking)
- `pytest-asyncio` (for async tests)

## Configuration

The test suite is configured with:
- **Async mode**: `asyncio_mode = auto` in `pytest.ini`
- **Coverage target**: 60% of core agent code
- **Automatic async test detection**: No need for explicit `@pytest.mark.asyncio` decorators

## Notes

- All AWS services are mocked in tests
- Async tests work automatically with `asyncio_mode = auto`
- Tests focus on the actual agent interface (not outdated methods)
- Coverage excludes non-core files for realistic metrics
- Lambda function tests include proper error handling validation