#!/usr/bin/env python3
"""
AI Domain Classifier

This script connects to the OpenRouter API to access the Gemini model,
evaluates .ai domains, and classifies them as either "sellable in the future"
or "marketable now" based on predefined criteria. The script will continue
generating domains until it finds 50 available ones.
"""

import os
import time
import json
import requests
import subprocess
import re
import functools
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add DNS resolver for faster domain checking
try:
    import dns.resolver
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False
    print("Warning: dnspython package not installed. Using slower WHOIS checks.")
    print("Install with: pip install dnspython")

# Configuration constants
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.0-flash-001"
TARGET_COUNT = 50
OUTPUT_FILE = "domains.txt"
MAX_ATTEMPTS = 1000

# Performance optimization settings
MAX_WORKERS = 10  # Maximum number of parallel threads
BATCH_SIZE = 5    # Number of domains to classify in one batch
WHOIS_TIMEOUT = 5  # Timeout for WHOIS checks in seconds
CACHE_SIZE = 1000  # Number of domains to cache results for

# Classification criteria constants
CLASSIFICATION_CRITERIA = """
Domain Classification Criteria:

1. Marketable Now:
   - Short domain (less than 8 characters)
   - Contains popular technology keywords (ai, ml, data, bot, etc.)
   - Relates to current AI trends (generative, llm, vision, agent, etc.)
   - Clear meaning and memorability
   - Generic or widely applicable in AI industry
   - Can be used for current AI products or services

2. Sellable in the Future:
   - Longer domain (8+ characters)
   - Relates to emerging or future AI concepts
   - Specific to niche AI applications
   - Contains forward-looking terminology
   - May be too specialized for immediate mass appeal
   - Value may increase as the AI field evolves
"""

class DomainClassifier:
    """Class to classify AI domains using the OpenRouter Gemma3 model."""
    
    def __init__(self):
        """Initialize the domain classifier with the hardcoded API key."""
        self.api_key = API_KEY
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://tldhunt.local"
        }
        # Initialize results cache
        self._availability_cache = {}
        self._classification_cache = {}
    
    @functools.lru_cache(maxsize=CACHE_SIZE)
    def check_domain_availability_dns(self, domain: str) -> bool:
        """
        Check if a domain is available using DNS resolution (faster than WHOIS).
        
        Args:
            domain: The domain to check
            
        Returns:
            True if the domain is available, False otherwise
        """
        if not DNS_AVAILABLE:
            return self.check_domain_availability_whois(domain)
            
        try:
            # Try to resolve the domain
            dns.resolver.resolve(domain, 'A')
            # If we get here, the domain exists (not available)
            return False
        except dns.resolver.NXDOMAIN:
            # NXDOMAIN means domain doesn't exist (available)
            return True
        except Exception as e:
            # Fall back to WHOIS on DNS errors
            print(f"DNS error for {domain}, falling back to WHOIS: {str(e)}")
            return self.check_domain_availability_whois(domain)
    
    @functools.lru_cache(maxsize=CACHE_SIZE)
    def check_domain_availability_whois(self, domain: str) -> bool:
        """
        Check if a domain is available using optimized WHOIS command.
        
        Args:
            domain: The domain to check
            
        Returns:
            True if the domain is available, False otherwise
        """
        try:
            # Run whois command with timeout to avoid hanging
            whois_output = subprocess.check_output(
                ["whois", domain], 
                stderr=subprocess.STDOUT,
                text=True,
                timeout=WHOIS_TIMEOUT
            )
            
            # Quick check for common availability indicators (faster than regex)
            if "No match for" in whois_output or "NOT FOUND" in whois_output:
                return True
            
            # Quick check for common registration indicators
            if ("Name Server:" in whois_output or "nameserver:" in whois_output or 
                "Domain Status:" in whois_output):
                return False
            
            # If no quick determination, use more comprehensive regex check
            registered_patterns = [
                r"Name Server", 
                r"nserver", 
                r"nameservers", 
                r"status: active",
                r"Domain Status: clientTransferProhibited",
                r"Domain Status: clientDeleteProhibited",
                r"Domain Status: clientRenewProhibited",
                r"Domain Status: clientUpdateProhibited",
            ]
            
            # Check for registration patterns
            for pattern in registered_patterns:
                if re.search(pattern, whois_output, re.IGNORECASE):
                    return False  # Domain is registered
            
            # If we reach here without a clear indication, we assume it might be available
            return True
            
        except subprocess.TimeoutExpired:
            print(f"WHOIS timeout for {domain}")
            return False
        except Exception as e:
            print(f"Error checking domain {domain}: {str(e)}")
            return False
    
    def check_domain_availability(self, domain: str) -> bool:
        """
        Check if a domain is available using the fastest available method.
        Uses caching to avoid duplicate checks.
        
        Args:
            domain: The domain to check
            
        Returns:
            True if the domain is available, False otherwise
        """
        # Check cache first
        if domain in self._availability_cache:
            return self._availability_cache[domain]
        
        # Prefer DNS check (faster) but fall back to WHOIS
        if DNS_AVAILABLE:
            result = self.check_domain_availability_dns(domain)
        else:
            result = self.check_domain_availability_whois(domain)
        
        # Cache the result
        self._availability_cache[domain] = result
        return result
    
    def check_domains_parallel(self, domains: List[str]) -> Dict[str, bool]:
        """
        Check multiple domains in parallel using ThreadPoolExecutor.
        
        Args:
            domains: List of domains to check
            
        Returns:
            Dictionary mapping domains to their availability (True=available)
        """
        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_domain = {executor.submit(self.check_domain_availability, domain): domain 
                                for domain in domains}
            
            for future in as_completed(future_to_domain):
                domain = future_to_domain[future]
                try:
                    results[domain] = future.result()
                except Exception as exc:
                    print(f"Domain check failed for {domain}: {exc}")
                    results[domain] = False
        
        return results
    
    def classify_domain(self, domain: str) -> Dict:
        """
        Classify an AI domain using Gemma3 through OpenRouter.
        
        Args:
            domain: The domain name to classify (without the TLD)
            
        Returns:
            Dictionary with classification results
        """
        # Check cache first
        if domain in self._classification_cache:
            return self._classification_cache[domain]
            
        full_domain = f"{domain}.ai"
        is_available = self.check_domain_availability(full_domain)
        
        # Prepare the prompt for Gemma3
        prompt = f"""
You are a domain name expert specializing in AI domains. Evaluate the domain name "{domain}.ai" 
and classify it as either "Marketable Now" or "Sellable in the Future" based on the following criteria:

{CLASSIFICATION_CRITERIA}

Respond with a JSON object with the following fields:
- "classification": either "Marketable Now" or "Sellable in the Future"
- "score": a score from 1-10 indicating the strength of the domain based on its classification (10 being highest)
- "reasoning": a brief explanation for your classification
- "potential_uses": 2-3 potential use cases for this domain
"""
        
        # Query the model
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": MODEL_NAME,  # Using the correct Gemma3 model via OpenRouter
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,  # Low temperature for more consistent results
                    "max_tokens": 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Try to parse the response as JSON
                try:
                    classification_data = json.loads(content)
                except json.JSONDecodeError:
                    # If JSON parsing fails, extract data with simple heuristics
                    classification = "Marketable Now" if "Marketable Now" in content else "Sellable in the Future"
                    score = 5  # Default score
                    reasoning = content
                    potential_uses = []
                    
                    classification_data = {
                        "classification": classification,
                        "score": score,
                        "reasoning": reasoning,
                        "potential_uses": potential_uses
                    }
                
                # Add availability information
                classification_data["available"] = is_available
                classification_data["domain"] = full_domain
                
                # Cache the result
                self._classification_cache[domain] = classification_data
                return classification_data
            else:
                result = {
                    "domain": full_domain,
                    "available": is_available,
                    "classification": "Error",
                    "score": 0,
                    "reasoning": f"API Error: {response.status_code}",
                    "potential_uses": []
                }
                self._classification_cache[domain] = result
                return result
                
        except Exception as e:
            result = {
                "domain": full_domain,
                "available": is_available,
                "classification": "Error",
                "score": 0,
                "reasoning": f"Exception: {str(e)}",
                "potential_uses": []
            }
            self._classification_cache[domain] = result
            return result
    
    def classify_domains_batch(self, domains: List[str]) -> List[Dict]:
        """
        Classify multiple domains in parallel using ThreadPoolExecutor.
        
        Args:
            domains: List of domains to classify
            
        Returns:
            List of classification results
        """
        results = []
        with ThreadPoolExecutor(max_workers=min(BATCH_SIZE, len(domains))) as executor:
            futures = {executor.submit(self.classify_domain, domain): domain for domain in domains}
            
            for future in as_completed(futures):
                domain = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Print result as it completes
                    if result["available"]:
                        print(f"✅ AVAILABLE! {result['domain']}")
                        print(f"Classification: {result['classification']}")
                        print(f"Score: {result['score']}/10")
                        print(f"Reasoning: {result['reasoning'][:100]}...")
                    else:
                        print(f"❌ Not available: {result['domain']}")
                except Exception as exc:
                    print(f"Domain classification failed for {domain}: {exc}")
                    results.append({
                        "domain": f"{domain}.ai",
                        "available": False,
                        "classification": "Error",
                        "score": 0,
                        "reasoning": f"Classification error: {str(exc)}",
                        "potential_uses": []
                    })
        
        return results
        
    def generate_domain_suggestions(self, num_domains: int = 30, existing_domains: List[str] = None) -> List[str]:
        """
        Generate new domain suggestions using Gemma3.
        
        Args:
            num_domains: Number of domains to generate
            existing_domains: List of domains already checked to avoid duplicates
            
        Returns:
            List of new domain suggestions
        """
        existing_domains_str = ""
        if existing_domains and len(existing_domains) > 0:
            existing_sample = existing_domains[:50] if len(existing_domains) > 50 else existing_domains
            existing_domains_str = "\n\nHere are some domains that have already been checked, please avoid suggesting these or similar ones:\n" + "\n".join(existing_sample)
        
        prompt = f"""As an AI domain expert, generate {num_domains} unique creative domain names for .ai domains.

Some will be "Marketable Now":
- Short (2-7 characters if possible)
- Contain tech keywords (ml, data, bot, etc.) - DO not include ai in the domain name, as the TLD is .ai
- Relate to current AI trends
- Clear meaning and memorability

Others will be "Sellable in the Future":
- Can be longer (8+ characters)
- Relate to emerging AI concepts
- Target niche AI applications
- Use forward-looking terminology

FORMAT INSTRUCTIONS (CRITICAL):
- List ONLY domain names without the .ai extension
- One domain per line
- No numbering, bullets, explanations, or other text
- Clean, simple words with no special characters
- Domains must contain only letters, numbers, and hyphens
- Examples of correctly formatted output:
  neural
  aismith
  deepthink
  botflow
  futureml
  cognilearn{existing_domains_str}"""
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "You are a domain name generation expert. You only respond with lists of domain names, one per line, with no additional formatting or text."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.6,  # Higher temperature for creativity
                    "max_tokens": 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse the domain names from the content
                lines = content.strip().split('\n')
                domains = []
                for line in lines:
                    # Clean up the line
                    domain = line.strip().lower()
                    # Remove any .ai extension if it was included
                    if domain.endswith('.ai'):
                        domain = domain[:-3]
                    # Remove any bullets, numbers, or other formatting
                    if domain and len(domain) > 0 and any(c.isalpha() for c in domain):
                        # Remove any non-alphanumeric characters except hyphen
                        domain = ''.join(c for c in domain if c.isalnum() or c == '-')
                        domains.append(domain)
                
                # Filter out any existing domains
                if existing_domains:
                    domains = [d for d in domains if d not in existing_domains]
                
                return domains
            else:
                print(f"Error generating domain suggestions: API returned status code {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Exception while generating domain suggestions: {str(e)}")
            return []

def main():
    """Main function to run the domain classifier."""
    print("========================================")
    print("  .ai Domain Finder powered by Gemma3")
    print("========================================")
    print()
    
    print(f"Target: {TARGET_COUNT} available domains")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Performance settings: {MAX_WORKERS} parallel workers, batch size {BATCH_SIZE}")
    print()
    print("Classification Criteria:")
    print("- 'Marketable Now': Short, current tech keywords, memorable")
    print("- 'Sellable in Future': Longer, emerging concepts, niche applications")
    print()
    
    # Initialize classifier
    classifier = DomainClassifier()
    
    # Track progress
    available_domains = []
    marketable_now = []
    sellable_future = []
    checked_domains = set()
    domain_results = []
    
    # Start with an empty list of domains to check
    domains_to_check = []
    
    attempt_count = 0
    generation_count = 0
    start_time = time.time()
    
    # Loop until we have the target number of available domains or reach max attempts
    while len(available_domains) < TARGET_COUNT and attempt_count < MAX_ATTEMPTS:
        # If we've checked all domains in our list or have few left, generate more
        if len(domains_to_check) < BATCH_SIZE:
            generation_count += 1
            print(f"\n=== Generating domain suggestions (Batch #{generation_count}) ===")
            
            # Generate new domain suggestions using Gemma3
            new_domains = classifier.generate_domain_suggestions(
                num_domains=50,  # Ask for more to account for filtering
                existing_domains=list(checked_domains)
            )
            
            if not new_domains:
                print("⚠️ Failed to generate domains. Waiting 3 seconds before retrying...")
                time.sleep(3)
                continue
            
            print(f"Generated {len(new_domains)} new domain suggestions")
            
            # Add new domains to the domains to check
            domains_to_check.extend([d for d in new_domains if d not in checked_domains])
        
        # Take a batch of domains to process in parallel
        batch = domains_to_check[:BATCH_SIZE]
        domains_to_check = domains_to_check[BATCH_SIZE:]
        
        # Skip domains we've already checked
        batch = [d for d in batch if d not in checked_domains]
        if not batch:
            continue
            
        attempt_count += len(batch)
        
        print(f"\nProcessing {attempt_count}/{MAX_ATTEMPTS} (Found: {len(available_domains)}/{TARGET_COUNT}): {', '.join([d+'.ai' for d in batch])}")
        
        # First, check availability in parallel (faster)
        print("Checking domain availability...")
        availability_results = classifier.check_domains_parallel({f"{d}.ai": d for d in batch})
        
        # Filter to available domains for classification
        available_batch = [d for d, domain in zip(batch, [f"{d}.ai" for d in batch]) if availability_results.get(domain, False)]
        
        if available_batch:
            print(f"Found {len(available_batch)} available domains. Classifying...")
            
            # Classify available domains in parallel
            batch_results = classifier.classify_domains_batch(available_batch)
            domain_results.extend(batch_results)
            
            # Process results
            for result in batch_results:
                if result["available"]:
                    available_domains.append(result["domain"])
                    
                    # Track classification
                    if result["classification"] == "Marketable Now":
                        marketable_now.append(result["domain"])
                    elif result["classification"] == "Sellable in the Future":
                        sellable_future.append(result["domain"])
        else:
            print("No available domains in this batch.")
        
        # Mark all as checked
        checked_domains.update(batch)
        
        # Short pause to avoid API throttling but much faster than before
        if len(available_domains) < TARGET_COUNT and attempt_count < MAX_ATTEMPTS:
            time.sleep(0.5)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    domains_per_second = attempt_count / elapsed_time if elapsed_time > 0 else 0
    
    # Summary
    print("\n===== DOMAIN SEARCH SUMMARY =====")
    print(f"Domains Checked: {attempt_count}")
    print(f"Available Domains Found: {len(available_domains)}/{TARGET_COUNT}")
    print(f"Marketable Now: {len(marketable_now)}")
    print(f"Sellable in the Future: {len(sellable_future)}")
    print(f"Time Elapsed: {elapsed_time:.2f} seconds")
    print(f"Processing Speed: {domains_per_second:.2f} domains/second")
    
    # Save available domains to output file
    with open(OUTPUT_FILE, 'w') as f:
        for domain in available_domains:
            f.write(f"{domain}\n")
    
    print(f"\nAvailable domains saved to {OUTPUT_FILE}")
    
    # Also save detailed results
    json_output = "domains_detailed.json"
    with open(json_output, 'w') as f:
        json.dump({
            "available_domains": available_domains,
            "marketable_now": marketable_now,
            "sellable_future": sellable_future,
            "domains_checked": list(checked_domains),
            "domain_results": domain_results,
            "classification_criteria": CLASSIFICATION_CRITERIA,
            "performance": {
                "elapsed_time": elapsed_time,
                "domains_per_second": domains_per_second,
                "total_domains_checked": attempt_count
            }
        }, f, indent=2)
    
    print(f"Detailed results saved to {json_output}")


if __name__ == "__main__":
    main() 