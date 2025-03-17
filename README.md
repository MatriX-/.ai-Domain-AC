# .ai Domain Availability & Classification Tool

This tool helps you discover and evaluate available .ai domain names by leveraging the power of LLMs (via OpenRouter) to generate creative domain name suggestions and classify them based on their market potential.

## Features

- **Domain Name Generation**: Automatically generates creative .ai domain name suggestions
- **Availability Checking**: Fast, parallel checking of domain availability using DNS and WHOIS
- **Domain Classification**: Classifies domains as either "Marketable Now" or "Sellable in the Future"
- **Scoring System**: Rates domains on a scale of 1-10 for their potential value
- **Detailed Analysis**: Provides reasoning and potential use cases for each domain
- **High Performance**: Optimized with parallel processing and caching for speed

## Requirements

- Python 3.6+
- OpenRouter API key (for accessing LLM models)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/.ai-Domain-AC.git
   cd .ai-Domain-AC
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

Run the script with:

```
python ai_domain_classifier.py
```

The script will:
1. Generate batches of domain name suggestions
2. Check their availability in parallel
3. Classify available domains using the specified LLM model
4. Continue until finding 50 available domains or reaching the maximum attempt limit

## Configuration

You can customize the script's behavior by modifying these variables:

- `TARGET_COUNT`: Number of available domains to find (default: 50)
- `MAX_ATTEMPTS`: Maximum number of domain checks (default: 1000)
- `MAX_WORKERS`: Number of parallel threads for checking (default: 10)
- `BATCH_SIZE`: Number of domains to process in batch (default: 5)
- `MODEL_NAME`: LLM model to use for classification (default: "google/gemini-2.0-flash-001")

## Output

The script generates two output files:
- `domains.txt`: Simple list of available domains
- `domains_detailed.json`: Detailed information including classifications, scores, and performance metrics

## Classification Criteria

Domains are classified based on:

### Marketable Now
- Short domain (less than 8 characters)
- Contains popular technology keywords
- Relates to current AI trends
- Clear meaning and memorability
- Generic or widely applicable in AI industry

### Sellable in the Future
- Longer domain (8+ characters)
- Relates to emerging or future AI concepts
- Specific to niche AI applications
- Contains forward-looking terminology
- Value may increase as the AI field evolves

## License

MIT