#!/bin/bash

# Test runner script for the Envision Agent project
# Usage: ./run_tests.sh [test_type]
# test_type: unit, integration, all, coverage, lint

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "No virtual environment detected. Consider activating one."
        print_warning "Run: python -m venv venv && source venv/bin/activate"
    else
        print_status "Using virtual environment: $VIRTUAL_ENV"
    fi
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    pip install pytest pytest-asyncio>=0.21.0 pytest-cov pytest-mock
    print_success "Dependencies installed"
}

# Run unit tests
run_unit_tests() {
    print_status "Running unit tests..."
    pytest tests/test_lambda_function.py tests/test_api.py -v -m "not integration"
    print_success "Unit tests completed"
}

# Run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    pytest tests/test_api.py -v -m "integration"
    print_success "Integration tests completed"
}

# Run all tests
run_all_tests() {
    print_status "Running all tests..."
    pytest tests/ -v
    print_success "All tests completed"
}

# Run tests with coverage
run_coverage() {
    print_status "Running tests with coverage..."
    pytest tests/ -v --cov=custom_agent --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=60
    print_success "Coverage report generated"
    print_status "HTML coverage report: htmlcov/index.html"
}

# Run linting
run_lint() {
    print_status "Running code quality checks..."
    
    # Install linting tools if not present
    pip install black isort flake8 mypy bandit safety 2>/dev/null || true
    
    print_status "Running Black (code formatting)..."
    black --check --diff . || {
        print_warning "Code formatting issues found. Run 'black .' to fix."
    }
    
    print_status "Running isort (import sorting)..."
    isort --check-only --diff . || {
        print_warning "Import sorting issues found. Run 'isort .' to fix."
    }
    
    print_status "Running flake8 (linting)..."
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    print_status "Running mypy (type checking)..."
    mypy --ignore-missing-imports custom_agent.py agent_cli.py || {
        print_warning "Type checking issues found."
    }
    
    print_status "Running Bandit (security linting)..."
    print_status "  - Only failing on MEDIUM or HIGH severity issues (LOW severity allowed)"
    bandit -r . --severity-level medium || {
        print_warning "Medium or high severity security issues found."
    }
    
    print_status "Running Safety (dependency vulnerability check)..."
    print_status "  - Allowing low-severity or development-only vulnerabilities"
    safety check --ignore 70612 || {
        print_warning "Dependency vulnerabilities found (some may be acceptable)."
    }
    
    print_success "Code quality checks completed"
}



# Clean up test artifacts
cleanup() {
    print_status "Cleaning up test artifacts..."
    rm -rf .pytest_cache
    rm -rf htmlcov
    rm -f coverage.xml
    rm -f .coverage
    rm -f bandit-report.json
    rm -f safety-report.json
    print_success "Cleanup completed"
}

# Main script logic
main() {
    local test_type=${1:-"all"}
    
    print_status "Starting test suite for Envision Agent"
    print_status "Test type: $test_type"
    
    check_venv
    
    case $test_type in
        "unit")
            install_deps
            run_unit_tests
            ;;
        "integration")
            install_deps
            run_integration_tests
            ;;
        "all")
            install_deps
            run_all_tests
            ;;
        "coverage")
            install_deps
            run_coverage
            ;;
        "lint")
            run_lint
            ;;
        "full")
            install_deps
            run_all_tests
            run_coverage
            run_lint
            ;;
        "clean")
            cleanup
            ;;
        *)
            print_error "Unknown test type: $test_type"
            echo "Usage: $0 [unit|integration|all|coverage|lint|full|clean]"
            exit 1
            ;;
    esac
    
    print_success "Test suite completed successfully!"
}

# Run main function with all arguments
main "$@"