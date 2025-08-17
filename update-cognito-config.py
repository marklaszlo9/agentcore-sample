#!/usr/bin/env python3
"""
Script to update Cognito configuration in frontend files
"""

import json
import re
import sys
from pathlib import Path


def load_config(config_file="cognito-config.json"):
    """Load Cognito configuration from JSON file"""
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON in {config_file}")
        sys.exit(1)


def update_auth_js(config):
    """Update auth.js with new configuration"""
    auth_file = Path("static-frontend/auth.js")

    if not auth_file.exists():
        print("auth.js not found")
        return False

    content = auth_file.read_text()

    # Update configuration object
    new_config = f"""const cognitoAuthConfig = {{
    authority: "https://cognito-idp.{config['region']}.amazonaws.com/{config['userPoolId']}",
    client_id: "{config['clientId']}",
    redirect_uri: "{config['redirectUri']}",
    response_type: "code",
    scope: "{' '.join(config['scopes'])}"
}};"""

    # Replace the configuration
    pattern = r"const cognitoAuthConfig = \{[^}]+\};"
    content = re.sub(pattern, new_config, content, flags=re.DOTALL)

    # Update logout function
    logout_function = f"""export async function signOutRedirect() {{
    const clientId = "{config['clientId']}";
    const logoutUri = "{config['logoutUri']}";
    const cognitoDomain = "https://{config['domain']}";
    
    window.location.href = `${{cognitoDomain}}/logout?client_id=${{clientId}}&logout_uri=${{encodeURIComponent(logoutUri)}}`;
}};"""

    pattern = r"export async function signOutRedirect\(\) \{[^}]+\};"
    content = re.sub(pattern, logout_function, content, flags=re.DOTALL)

    auth_file.write_text(content)
    print("✅ Updated auth.js")
    return True


def main():
    """Main function"""
    config_file = sys.argv[1] if len(sys.argv) > 1 else "cognito-config.json"

    print(f"Loading configuration from {config_file}...")
    config = load_config(config_file)

    print("Updating frontend files...")
    update_auth_js(config)

    print("✅ Configuration update complete!")
    print(f"Frontend will use:")
    print(f"  - User Pool: {config['userPoolId']}")
    print(f"  - Client ID: {config['clientId']}")
    print(f"  - Redirect URI: {config['redirectUri']}")


if __name__ == "__main__":
    main()
