import sys
import os
import runpy

# Add the application's root directory to the Python path.
# This is necessary to ensure that local modules can be found,
# especially when running inside a container or with wrappers like
# 'opentelemetry-instrument' that can affect Python's default path.
# The working directory is set to /app in the Dockerfile.
sys.path.insert(0, '/app')

# Execute the main application module.
# Using runpy.run_module is equivalent to running 'python -m runtime_agent_main'
# but allows us to set up the path first.
runpy.run_module('runtime_agent_main', run_name='__main__')
