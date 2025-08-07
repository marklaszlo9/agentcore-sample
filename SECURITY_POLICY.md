# Security Policy

## Security Scanning Configuration

This project uses automated security scanning to ensure code safety while maintaining development velocity.

### Security Severity Levels

Our CI/CD pipeline is configured with the following security policy:

#### ✅ **Passing Conditions:**
- **LOW severity issues** - Allowed to pass (warnings only)
- **No security issues** - Passes

#### ❌ **Failing Conditions:**
- **MEDIUM severity issues** - Fails the build
- **HIGH severity issues** - Fails the build

### Tools Used

#### **Bandit (Static Code Analysis)**
- Scans Python code for common security issues
- Configuration: `--severity-level medium`
- Only fails on medium or high severity findings

#### **Safety (Dependency Vulnerability Check)**
- Scans dependencies for known vulnerabilities
- Allows low-severity or development-only vulnerabilities
- Configured with specific ignores for acceptable risks

### Local Testing

Run security checks locally:

```bash
# Run all security checks
./run_tests.sh lint

# Run only security scans
bandit -r . --severity-level medium
safety check --ignore 70612
```

### Security Best Practices Implemented

1. **Secure Network Binding**
   - Default: Binds to `127.0.0.1` (localhost only)
   - Container: Uses `HOST` environment variable for flexibility
   - No hardcoded binding to all interfaces (`0.0.0.0`)

2. **Dependency Management**
   - Regular dependency vulnerability scanning
   - Pinned versions in `requirements.txt`
   - Automated security updates via CI/CD

3. **Code Quality**
   - Static analysis with Bandit
   - Type checking with mypy
   - Code formatting with Black and isort

### Reporting Security Issues

If you discover a security vulnerability, please:

1. **Do not** create a public GitHub issue
2. Contact the maintainers directly
3. Provide detailed information about the vulnerability
4. Allow time for the issue to be addressed before public disclosure

### Security Updates

- Security scans run on every pull request
- Dependencies are regularly updated
- Security patches are prioritized for immediate deployment