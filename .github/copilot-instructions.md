# Copilot Instructions for Motor TPL Pricing Engine

## Project Overview
This is a Python-based pricing engine for Motor Third Party Liability (TPL) insurance. The system calculates premiums based on vehicle characteristics, driver profiles, and risk factors.

## Architecture
- **Core Components**: Pricing models, risk assessment, data processing
- **Data Flow**: Input validation → Risk calculation → Premium computation → Output formatting
- **Key Patterns**: Modular design with separate concerns for data ingestion, processing, and output

## Development Workflow
- **Environment**: Use the provided virtual environment (.venv)
- **Dependencies**: Managed via pyproject.toml (PEP 621 compliant)
- **Build System**: Makefile for common tasks (when implemented)
- **Testing**: [To be defined - add when test framework is chosen]

## Code Conventions
- **Python Version**: [Specify in pyproject.toml when set]
- **Style**: Follow PEP 8 with [linter/formatter when chosen]
- **Imports**: [Define import organization patterns when code exists]
- **Error Handling**: [Document specific error patterns when implemented]

## Key Files and Directories
- `pyproject.toml`: Project configuration and dependencies
- `Makefile`: Build and development tasks
- `README.md`: Project documentation
- `.venv/`: Virtual environment (ignored in .gitignore)

## Integration Points
- **External Dependencies**: [List when added, e.g., data sources, ML libraries]
- **APIs**: [Define API patterns when implemented]
- **Data Sources**: [Document data integration patterns]

## Common Patterns
- [Add specific patterns as code is developed, e.g., pricing calculation formulas, data validation rules]

## Notes for AI Agents
- Focus on modular, testable code for pricing logic
- Prioritize accuracy and performance in calculations
- Document assumptions in risk models clearly
- Use type hints for data structures