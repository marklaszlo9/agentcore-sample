# agent_cli.py
import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from custom_agent import CustomEnvisionAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-cli")


async def main():
    parser = argparse.ArgumentParser(description="AgentCore Custom Agent CLI")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument(
        "--model-id",
        type=str,
        default="us.amazon.nova-micro-v1:0",
        help="Bedrock model ID to use",
    )
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument(
        "--kb-id", type=str, help="Knowledge base ID (overrides environment variable)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Get knowledge base ID from args or environment
    knowledge_base_id = args.kb_id or os.environ.get("STRANDS_KNOWLEDGE_BASE_ID")

    if not knowledge_base_id:
        logger.warning("No knowledge base ID provided. Agent will work without RAG.")

    try:
        # Create custom agent
        logger.info("Creating CustomEnvisionAgent...")
        agent = CustomEnvisionAgent(
            model_id=args.model_id,
            region=args.region,
            knowledge_base_id=knowledge_base_id,
            memory_id=os.environ.get("AGENTCORE_MEMORY_ID"),
        )

        if args.query:
            # Single query mode
            logger.info(f"Processing single query: {args.query}")
            if knowledge_base_id:
                response = await agent.query_with_rag(args.query)
            else:
                response = await agent.query(args.query)

            extracted_text = agent.extract_text_from_response(response)
            print(f"\nAgent: {extracted_text}")
        else:
            # Interactive chat loop
            print(agent.get_initial_greeting())
            logger.info("Starting interactive chat session. Type 'exit' to quit.")

            while True:
                user_input = input("\nQuery: ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                if knowledge_base_id:
                    response = await agent.query_with_rag(user_input)
                else:
                    response = await agent.query(user_input)

                extracted_text = agent.extract_text_from_response(response)
                print(f"\nAgent: {extracted_text}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()


def cli():
    """Command-line interface for the AgentCore Custom Agent"""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
