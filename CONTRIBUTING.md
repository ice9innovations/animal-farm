# Contributing to Animal Farm

Thank you for your interest in contributing to the Animal Farm ML platform! This document outlines the guidelines and standards that all contributors must follow.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Standards](#development-standards)
- [Data Protection Rules](#data-protection-rules)
- [Common Pitfalls](#common-pitfalls)
- [Submitting Changes](#submitting-changes)
- [AI Assistant Guidelines](#ai-assistant-guidelines)

## Code of Conduct

### Core Principles
1. **Respect the data** - Never modify data formats without explicit permission
2. **Follow specifications exactly** - No unsolicited "improvements" or "optimizations"
3. **Be honest about limitations** - Don't promise solutions you can't deliver
4. **Protect expensive resources** - ML processing time and GPU compute are valuable

## Getting Started

### Required Reading
Before making any changes, please read:
- This entire CONTRIBUTING.md file
- The `README.md` in the root directory
- Any relevant service-specific README files

### Understanding the Architecture
Animal Farm is a complex ML platform with multiple services. Take time to understand:
- The existing data flow patterns
- Why certain architectural decisions were made
- The difference between real-time API responses and batch processing formats

## Development Standards

### Best Practices

We encourage modern development practices that lead to reliable, maintainable code:

#### Test-Driven Development (TDD)
- Write tests before implementing features when possible
- Include test cases for edge conditions and error scenarios
- Ensure tests can be run independently and repeatedly
- Use descriptive test names that explain the expected behavior

#### Fail Fast Philosophy
- **Fail early** - Validate inputs and assumptions at the start
- **Fail often** - Better to catch problems during development than production
- **Fail loudly** - Clear error messages help debug issues quickly
- Use assertions and input validation to catch problems immediately

#### Clean Code Practices
- **Single Responsibility** - Each function should do one thing well
- **DRY (Don't Repeat Yourself)** - Extract common logic into reusable functions
- **KISS (Keep It Simple, Stupid)** - Simple solutions are easier to understand and maintain
- **YAGNI (You Aren't Gonna Need It)** - Don't build features until they're actually needed

#### Documentation as Code
- Write self-documenting code with clear variable and function names
- Include docstrings for complex functions
- Update documentation when you change functionality
- Leave comments explaining *why* something is done, not just *what* is done

#### Incremental Development
- Make small, focused changes rather than large refactors
- Test each change before moving to the next
- Commit early and often with descriptive messages
- Build on existing patterns rather than inventing new ones

### Core Rules

#### Rule #1: Respect the Data
```diff
- NEVER modify data formats without explicit permission
- NEVER add "optimizations" or "improvements" to data storage
- NEVER use formatter classes unless specifically requested
+ When asked to "save API data to database" - save it EXACTLY as received
+ Ask "Should I modify the data format?" before any transformation
```

#### Rule #2: Follow Specifications Exactly
```diff
- Don't add features nobody requested
- Don't "improve" working systems
- Don't create abstractions unless explicitly asked
+ Read requirements literally
+ If the task is "copy A to B" - just copy A to B
+ Keep it simple and direct
```

#### Rule #3: Debug Systematically
```diff
- Don't blame external systems or environments
- Don't make assumptions without evidence
- Don't skip obvious checks
+ When something doesn't work, check your own code first
+ Look for syntax errors, logic problems, or missing dependencies
+ Test systematically from simple to complex
```

#### Rule #4: Be Honest About Limitations
```diff
- Don't waste time on approaches that won't work
- Don't present theories as facts
- Don't promise solutions you can't deliver
+ If you don't know something, research it or ask
+ If something won't work, explain why clearly
+ Provide alternative approaches when current approach fails
```

## Data Protection Rules

### Critical: Expensive Resources
- **Hours of ML processing are expensive** - don't waste computational resources
- **GPU time costs money** - don't break optimized workflows
- **Development time is valuable** - don't create unnecessary work
- **Large datasets are irreplaceable** - don't corrupt processed data
- **Production systems serve users** - don't break working functionality

### Data Handling
- Always preserve original data formats unless explicitly asked to change them
- Never apply transformations without permission
- When storing API responses, store them exactly as received
- Always ask before any data transformation: "Should I modify this data?"

### Examples of Protected Data
- COCO dataset processing results (118K images, 15+ hours of compute)
- ML service raw outputs
- Database records from production systems
- User uploaded images and metadata

## Common Pitfalls

### Anti-Patterns to Avoid

#### Pattern 1: Overconfidence
```diff
- Making bold claims without investigation
- Assuming solutions will work without testing
+ Verify assumptions before implementing
+ Test incrementally
```

#### Pattern 2: Creative Solutions
```diff
- Inventing complex solutions instead of following instructions
- Adding undocumented features or parameters
+ Follow the exact specifications given
+ Don't add creative interpretations
```

#### Pattern 3: Repetitive Mistakes
```diff
- Making the same errors multiple times
- Ignoring feedback from previous attempts
+ Learn from immediate feedback
+ Keep track of what's already been tried
```

#### Pattern 4: Theory Over Evidence
```diff
- Creating elaborate theories about problems
- Blaming architecture, GPU, or external factors
+ Look at the actual code first
+ Check for simple, obvious problems
```

#### Pattern 5: Accumulating Technical Debt
```diff
- Creating layers of solutions without cleanup
- Leaving behind broken attempts
+ Clean up failed attempts
+ Build on existing work, don't replace it
```

### Technical Anti-Patterns
- **Data transformation syndrome**: Silently transforming data formats
- **Optimization obsession**: Adding unnecessary performance improvements
- **Abstraction addiction**: Creating classes and services nobody requested
- **Debug spam**: Adding excessive logging instead of fixing root cause
- **Blame shifting**: Assuming problems are external
- **Unsolicited additions**: Adding marketing claims, legal statements, or editorial commentary without request
- **Performance degradation**: Suggesting CPU-only mode as a solution to GPU problems

## Submitting Changes

### Before You Submit
1. **Test your changes** - Make sure they work as expected
2. **Write tests** - Include unit tests for new functionality when possible
3. **Verify data integrity** - Ensure no data corruption or unwanted transformation
4. **Check for breaking changes** - Don't break existing functionality
5. **Validate inputs** - Ensure your code fails fast with clear error messages
6. **Clean up** - Remove any debug code or temporary files
7. **Document** - Update relevant documentation if needed
8. **Follow DRY principles** - Extract any duplicated logic into reusable functions

### Pull Request Guidelines
- Describe what you changed and why
- Include steps to test the changes
- Note any data format changes (these need special approval)
- Reference any issues or requirements addressed

### Code Review Process
All changes go through code review to ensure:
- Adherence to these guidelines
- No data corruption or unwanted transformations
- Proper error handling and testing
- Documentation is updated

## Getting Help

### If You're Stuck
- Ask for help instead of guessing
- Admit when you don't know something
- Focus on the actual problem, not theoretical improvements

### Resources
- `docs/` directory contains extensive documentation
- Service-specific README files
- Existing code patterns show the preferred approaches

## AI Assistant Guidelines

*This project uses AI assistance for development. The following guidelines are specifically for AI contributors to prevent common AI-specific failure patterns.*

### Critical AI-Specific Rules

#### Never Hallucinate Technical Details
- **Don't invent port numbers** - Humans don't randomly guess ports, neither should you
- **Don't make up command line arguments** - Check documentation or ask
- **Don't fabricate API endpoints** - Verify actual endpoints exist
- **Don't assume file paths** - Check if files/directories actually exist

#### Behavioral Guidelines

**Danger Signs - Stop Immediately If You:**
- Start blaming the user's basic competencies
- Create elaborate theories without checking simple things
- Add multiple layers of "final solutions"
- Modify data formats without permission
- Promise solutions you know won't work
- Lose track of the original problem
- Create more files than the user asked for
- Debug your own debug output

**Communication Standards:**
- Be direct and concise
- Listen to feedback carefully
- When the user says "stop" - stop immediately
- When the user says "this is wrong" - it's wrong
- Don't argue with the user about their own requirements

**Debugging Partnership Principles:**
- **Trust user expertise**: When user says "this server works fine for everything else" - believe them
- **Focus on differences**: Look at what's unique about the failing case, not theoretical system problems
- **Question your assumptions first**: Before questioning the user's environment, question your own understanding

### AI Debugging Guidelines

#### Error Message Analysis
When debugging mysterious errors, remember that error messages can have complex histories:

- **Don't assume HTTP errors are current**: Generic messages like "Request failed with status code 404" may be stored database records from previous failed operations, not current server errors
- **Check error sources**: Previous AI instances may have stored axios/HTTP client error messages as database records
- **Verify error origin**: Always confirm whether error messages are coming from current requests or historical data

#### Domain/URL Debugging
Simple mistakes in URLs cause disproportionate debugging time:

- **Check for domain typos**: Verify hyphens vs dots (window-to-the-world-org vs window-to-the-world.org)
- **Verify protocols**: Ensure http vs https matches expectations
- **Trace actual requests**: Don't assume routing issues when it might be a malformed URL
- **Relative vs absolute paths**: Check if relative URLs are resolving to expected domains

#### Credentials and Security (Critical)
Security issues must be addressed immediately, never deferred:

- **Flag hardcoded credentials immediately**: Even in "demo" or "internal" files - they can be accidentally committed
- **Replace with environment variables**: Don't wait for "later" - do it during the current session
- **Audit beyond the immediate file**: Check for credential exposure in related files
- **Understand the blast radius**: Database credentials, API keys, etc. can compromise entire systems in seconds

#### System-Specific Issues

**PHP-Specific Gotchas:**
Common PHP compatibility issues that masquerade as other problems:

- **PHP version differences**: PHP 7 vs PHP 8 have different error handling, especially with MongoDB date objects
- **MongoDB date handling**: Calling `->toDateTime()` on null/undefined MongoDB date fields causes fatal errors
- **Connection string formats**: Duplicated query parameters in MongoDB connection strings cause connection failures
- **Error visibility**: PHP fatal errors may not appear in standard error logs, causing mysterious 500 errors

### The Meta-Rule for AI Assistants
**When user expertise conflicts with your theory, trust the user first, debug your assumptions second.**

The user knows their system, their requirements, and their data better than you do. Your job is to help them achieve their goals, not to impose your theories about what might be wrong.

## Final Notes

This project has suffered from well-intentioned but destructive "improvements" that have cost hundreds of hours and thousands of dollars in wasted compute resources. **Every rule in this document was written because someone violated it and caused real damage.**

The goal is not to restrict creativity but to ensure that contributions actually help rather than harm. When in doubt, ask. When asked to do something specific, do exactly that. When you encounter limitations, be honest about them.

**Remember**: The best contribution is often the one that solves the exact problem requested without creating new problems in the process.

### Positive Examples

Here are examples of good contributions we encourage:

#### Good: Input Validation
```python
def process_image(image_path):
    if not image_path:
        raise ValueError("image_path is required")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    # ... rest of function
```

#### Good: Clear Error Messages
```python
try:
    result = ml_service.analyze(image_url)
except ConnectionError as e:
    raise ConnectionError(f"Failed to connect to ML service at {service_url}: {e}")
```

#### Good: Test-First Development
```python
def test_save_analysis_preserves_original_format():
    """Test that saving analysis data preserves the original API response format"""
    original_data = {"predictions": [{"label": "cat", "confidence": 0.95}]}
    saved_data = save_analysis_to_db(original_data)
    assert saved_data["analysis"] == original_data
```

#### Good: Building on Existing Patterns
```python
# Follow existing naming conventions
# Use existing error handling patterns
# Integrate with existing logging system
```

**Remember**: Code that fails fast with clear messages is infinitely better than code that silently corrupts data or produces mysterious errors hours later.

---

*This document exists to protect both the project and contributors from repeating documented failure patterns. Following these guidelines ensures your contributions will be valued and effective.*