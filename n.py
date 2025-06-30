from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import aiohttp
import asyncio
import logging
from cachetools import TTLCache
import re
import json
from uuid import uuid4
from bs4 import BeautifulSoup
import html2text
import time
from firecrawl import FirecrawlApp
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from duckduckgo_search import DDGS 
from linkedin_scraper import actions

# Configure logging
import httpx
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EngagementTracker(BaseModel):
    last_contact: Optional[datetime] = None
    response_status: str = "not_contacted"
    follow_up_count: int = 0

class Candidate(BaseModel):
    name: str
    linkedin_url: str
    headline: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    education: List[Dict[str, str]] = Field(default_factory=list)
    experience: List[Dict[str, str]] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    about: Optional[str] = None

class ScoreBreakdown(BaseModel):
    education: float
    experience: float
    skills: float
    company_fit: float
    location: float
    tenure: float

class ScoredCandidate(BaseModel):
    candidate: Candidate
    fit_score: float
    score_breakdown: ScoreBreakdown
    outreach_message: Optional[str] = None

class JobConfig(BaseModel):
    company_name: str
    required_skills: List[str]
    nice_to_have_skills: List[str] = []
    location: str
    experience_years: int
    salary_range: str

class JobDescription(BaseModel):
    description: str
    config: JobConfig

class RawJobDescription(BaseModel):
    description: str


class LLMService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.models = [
            ("groq", "llama3-8b-8192", "https://api.groq.com/openai/v1/chat/completions"),
            ("groq", "mixtral-8x7b-32768", "https://api.groq.com/openai/v1/chat/completions"),
            ("openrouter", "anthropic/claude-3-sonnet", "https://openrouter.ai/api/v1/chat/completions"),
            ("openrouter", "meta-llama/llama-3-70b-instruct", "https://openrouter.ai/api/v1/chat/completions")
        ]
        self.rate_limits = {
            "groq": {"requests": 30, "window": 60},
            "openrouter": {"requests": 10, "window": 60}
        }
        self.locks = {k: asyncio.Lock() for k in self.rate_limits}
        self.request_counts = {k: [] for k in self.rate_limits}

    async def query_llm(self, prompt: str, model_index: int = 0, json_mode: bool = False) -> Optional[str]:
        provider, model, url = self.models[model_index % len(self.models)]
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY') if provider == 'groq' else os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        now = datetime.now()
        async with self.locks[provider]:
            self.request_counts[provider] = [t for t in self.request_counts[provider] if (now - t) < timedelta(seconds=self.rate_limits[provider]["window"])]
            if len(self.request_counts[provider]) >= self.rate_limits[provider]["requests"]:
                wait_time = self.rate_limits[provider]["window"] - (now - self.request_counts[provider][0]).seconds
                logger.warning(f"Rate limit for {provider} reached. Waiting for {wait_time} seconds.")
                await asyncio.sleep(wait_time)
            self.request_counts[provider].append(datetime.now())
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2 if json_mode else 0.7,
            "max_tokens": 2048 if json_mode else 1000
        }
        
        # Use JSON mode for providers that support it to get reliable JSON output
        if json_mode and provider in ['groq', 'openrouter']:
            if "claude" not in model: # Claude doesn't use response_format
                data["response_format"] = {"type": "json_object"}

        try:
            response = await self.client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
            logger.error(f"API request failed for {provider}: {str(e)}")
            if model_index < len(self.models) - 1:
                return await self.query_llm(prompt, model_index + 1)
            return None

    async def extract_job_config(self, job_description: str) -> Optional[JobConfig]:
        """Uses an LLM to extract structured data from a raw job description."""
        prompt = f"""
        Analyze the following job description and extract the information into a JSON object matching the provided schema.

        Job Description:
        ---
        {job_description}
        ---

        JSON Schema to populate:
        {{
            "company_name": "string",
            "required_skills": ["string"],
            "nice_to_have_skills": ["string"],
            "location": "string (e.g., 'Mountain View, CA')",
            "experience_years": "integer (minimum years required)",
            "salary_range": "string (e.g., '$150k - $200k USD')"
        }}

        Instructions:
        - Identify the single most important location.
        - For experience_years, find the minimum number of years required. If a range is given (e.g., 5-7 years), use the lower number.
        - Distinguish between essential 'required_skills' and 'nice_to_have_skills'.
        - If a piece of information is not present, use a sensible default (e.g., empty string, empty list, 0 for years).
        - Only return the JSON object, with no other text or explanations.
        """
        response_str = await self.query_llm(prompt, json_mode=True)
        if not response_str:
            return None
        try:
            data = json.loads(re.search(r'\{.*\}', response_str, re.DOTALL).group(0))
            return JobConfig(**data)
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.error(f"Failed to parse LLM response for job config: {e}")
            return None

class LinkedInSourcingAgent:
    def __init__(self):
        self.llm = LLMService()
        self.cache = TTLCache(maxsize=100, ttl=3600)
        self.session = None
        self.firecrawl_client = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY')
        self.rapidapi_host = "fresh-linkedin-profile-data.p.rapidapi.com"
        self.scrapingbee_api_key = os.getenv('SCRAPINGBEE_API_KEY')
        self.linkedin_cookie = os.getenv('LINKEDIN_LI_AT_COOKIE')
        self.linkedin_email = os.getenv('LINKEDIN_EMAIL')
        self.linkedin_password = os.getenv('LINKEDIN_PASSWORD')
        self.driver = None
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.google_cx = os.getenv('GOOGLE_SEARCH_CX')
        self.engagement_db = TTLCache(maxsize=500, ttl=timedelta(days=30))
        self.competitor_insights = {}

    async def initialize_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def initialize_driver(self):
        """Initializes the Selenium WebDriver and logs into LinkedIn."""
        if self.driver is None and self.linkedin_email and self.linkedin_password:
            logger.info("Initializing Selenium WebDriver and logging into LinkedIn...")
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._initialize_and_login_driver)
                logger.info("Successfully initialized WebDriver and logged in.")
            except Exception as e:
                logger.error(f"Failed to initialize WebDriver or login: {e}")
                self.driver = None # Ensure driver is None if init fails

    def _initialize_and_login_driver(self):
        """Synchronous helper to initialize driver and login."""
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = ChromeService(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        actions.login(self.driver, self.linkedin_email, self.linkedin_password)

    async def close_driver(self):
        if self.driver:
            logger.info("Closing Selenium WebDriver.")
            await asyncio.to_thread(self.driver.quit)
            self.driver = None

    async def _search_duckduckgo(self, query: str) -> List[Dict]:
        """Search for LinkedIn profiles using DuckDuckGo as a fallback."""
        logger.info(f"Attempting DuckDuckGo search with query: {query}")
        try:
            def search_sync():
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=20):
                        if 'linkedin.com/in/' in r.get('href', ''):
                            results.append({
                                'link': r['href'],
                                'title': r['title'],
                                'snippet': r['body']
                            })
                return results[:10]
            
            profiles = await asyncio.to_thread(search_sync)
            logger.info(f"Found {len(profiles)} potential profiles via DuckDuckGo.")
            return profiles
        except ImportError:
            logger.warning("`duckduckgo-search` library not installed. Skipping DuckDuckGo search. Run `pip install duckduckgo-search`.")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e.__class__.__name__}: {str(e)}")
            return []

    async def search_linkedin(self, job_description: str, config: JobConfig) -> List[Candidate]:
        cache_key = f"search_{hash(job_description)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        await self.initialize_session()
        search_query = await self._generate_search_query(config)
        logger.info(f"Searching with query: {search_query}")
        
        profiles = await self._search_google(search_query)

        if not profiles:
            logger.warning("Google Search returned no results. Falling back to DuckDuckGo Search.")
            profiles = await self._search_duckduckgo(search_query)

        candidates = []
        
        for profile in profiles[:10]:  # Limit to first 10 profiles
            linkedin_url = profile.get('link', '')
            if not linkedin_url.startswith('https://www.linkedin.com/in/'):
                continue
                
            # Check cache for existing profile
            profile_cache_key = f"profile_{hash(linkedin_url)}"
            if profile_cache_key in self.cache:
                candidates.append(self.cache[profile_cache_key])
                continue
                
            candidate = await self._fetch_candidate_data(linkedin_url, profile)
            if candidate:
                candidates.append(candidate)
                self.cache[profile_cache_key] = candidate
            await asyncio.sleep(3) # Add a 1-second delay to respect API rate limits

        self.cache[cache_key] = candidates
        return candidates

    async def _generate_search_query(self, config: JobConfig) -> str:
        base_query = f'site:linkedin.com/in/ ("{config.company_name}" OR "{config.location}")'
        skills_query = ' OR '.join(f'"{skill}"' for skill in config.required_skills)
        return f"{base_query} ({skills_query})"

    async def _search_google(self, query: str) -> List[Dict]:
        if not self.google_api_key or not self.google_cx:
            logger.warning("Google API credentials not configured - using mock data")
            return [
                {
                    "link": "https://www.linkedin.com/in/danmohd", 
                    "title": "Mohammed Danish - Acoustics Machine Learning Engineer - Apple",
                    "snippet": "Passionate about audio processing and machine learning. Building next-generation audio experiences at Apple. Skills: Python, TensorFlow, PyTorch.",
                    "pagemap": { "person": [{"role": "Acoustics Machine Learning Engineer", "org": "Apple", "name": "Mohammed Danish"}]}
                },
                {
                    "link": "https://www.linkedin.com/in/samcohan", 
                    "title": "Sam Cohan - Staff Software Engineer - Meta",
                    "snippet": "Working on large-scale distributed systems at Meta. Interested in reliability and performance.",
                    "pagemap": { "person": [{"role": "Staff Software Engineer", "org": "Meta", "name": "Sam Cohan"}]}
                },
                {
                    "link": "https://www.linkedin.com/in/surajkothawade", 
                    "title": "Suraj Kothawade - Software Engineer - Google",
                    "snippet": "Developing innovative solutions for Google Cloud. Focus on scalability and developer tools.",
                    "pagemap": { "person": [{"role": "Software Engineer", "org": "Google", "name": "Suraj Kothawade"}]}
                },
                {
                    "link": "https://www.linkedin.com/in/manjeetchhabra", 
                    "title": "Manjeet Singh Chhabra - Mountain View, California, United States",
                    "snippet": "Experienced engineering leader with a background in building and scaling teams."
                }
            ]

        params = {
            'key': self.google_api_key,
            'cx': self.google_cx,
            'q': query,
            'num': 10
        }

        try:
            async with self.session.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('items', [])
                logger.error(f"Google search failed with status: {response.status}")
                return []
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
            return []

    async def _fetch_linkedin_profile(self, linkedin_url: str, google_profile: Dict) -> Optional[Dict]:
        """Fetch LinkedIn profile using multiple sources with fallback"""
        # New: Try linkedin_scraper first as it's the most reliable when configured
        linkedin_scraper_data = await self._fetch_via_linkedin_scraper(linkedin_url)
        if linkedin_scraper_data and linkedin_scraper_data.get('data'):
            logger.info(f"Successfully fetched data for {linkedin_url} via Selenium + LLM Extraction.")
            return linkedin_scraper_data

        # Try RapidAPI first
        rapidapi_data = await self._fetch_via_rapidapi(linkedin_url)
        if rapidapi_data and rapidapi_data.get('data'):
            logger.info(f"Successfully fetched data for {linkedin_url} via RapidAPI.")
            return rapidapi_data
            
        # Fallback to Firecrawl
        firecrawl_data = await self._fetch_via_firecrawl(linkedin_url)
        if firecrawl_data:
            logger.info(f"Successfully fetched data for {linkedin_url} via Firecrawl.")
            return firecrawl_data

        # New Fallback to ScrapingBee
        scrapingbee_data = await self._fetch_via_scrapingbee(linkedin_url)
        if scrapingbee_data:
            logger.info(f"Successfully fetched data for {linkedin_url} via ScrapingBee.")
            return scrapingbee_data
            
        # Final fallback to mock data
        logger.warning(f"All scraping services failed for {linkedin_url} - using mock data.")
        return self._generate_mock_profile(linkedin_url, google_profile)

    async def _fetch_via_rapidapi(self, linkedin_url: str) -> Optional[Dict]:
        url = "https://fresh-linkedin-profile-data.p.rapidapi.com/get-profile-pdf-cv"
        params = {"linkedin_url": linkedin_url}
        headers = {
            "x-rapidapi-key": self.rapidapi_key,
            "x-rapidapi-host": self.rapidapi_host
        }

        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._normalize_rapidapi_data(data)
                error_data = await response.text()
                logger.error(f"RapidAPI request failed with status: {response.status}. Response: {error_data[:200]}")
                return None
        except Exception as e:
            logger.error(f"RapidAPI request failed: {str(e)}")
            return None

    async def _fetch_via_linkedin_scraper(self, linkedin_url: str) -> Optional[Dict]:
        """
        Scrape LinkedIn using Selenium for navigation and an LLM for robust data extraction.
        This bypasses the brittle parsing of the linkedin_scraper library.
        """
        if not self.driver:
            return None
        
        logger.info(f"Attempting to scrape {linkedin_url} via Selenium + LLM Extraction.")
        try:
            def get_page_source():
                self.driver.get(linkedin_url)
                time.sleep(5)  # Wait for dynamic content to load
                return self.driver.page_source

            loop = asyncio.get_running_loop()
            html_content = await loop.run_in_executor(None, get_page_source)

            if not html_content:
                logger.warning(f"Got empty page source from Selenium for {linkedin_url}")
                return None

            text_content = html2text.html2text(html_content)

            extraction_prompt = f"""
            Analyze the following text content from a LinkedIn profile and extract the information into a JSON object.
            The text is messy, do your best to find the relevant information.

            Text Content:
            ---
            {text_content[:15000]}
            ---

            JSON Schema to populate:
            {{
                "full_name": "string", "headline": "string", "current_company": "string",
                "location": "string", "about": "string",
                "experience": [{{ "title": "string", "company": "string", "duration": "string" }}],
                "education": [{{ "school": "string", "degree": "string", "duration": "string" }}],
                "skills": ["string"]
            }}

            Instructions:
            - For 'current_company', use the company from the most recent experience entry.
            - If a piece of information is not present, use a sensible default (e.g., empty string, empty list).
            - Only return the JSON object, with no other text or explanations.
            """

            response_str = await self.llm.query_llm(extraction_prompt)
            if not response_str: return None

            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                return {"data": json.loads(json_match.group(0))}
            
            logger.error(f"Failed to extract JSON from LLM response for {linkedin_url}")
            return None

        except Exception as e:
            logger.error(f"Selenium + LLM scraping failed for {linkedin_url}. Error: {e.__class__.__name__}: {str(e)}")
            return None

    async def _fetch_via_firecrawl(self, linkedin_url: str) -> Optional[Dict]:
        """Scrape LinkedIn profile using Firecrawl"""
        if not self.firecrawl_client.api_key: return None
        try:
            page_options = {
                "waitFor": "main.scaffold-layout__main",
                "timeout": 60000,  # 60 seconds timeout
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                }
            }
            if self.linkedin_cookie:
                logger.info("Using LinkedIn session cookie for Firecrawl.")
                page_options["cookies"] = [{"name": "li_at", "value": self.linkedin_cookie, "domain": ".linkedin.com"}]

            response = await self.firecrawl_client.scrape_url(
                url=linkedin_url,
                params={
                    "pageOptions": page_options,
                    "extractorOptions": {
                        "mode": "llm-extraction",
                        "extractionSchema": {
                            "type": "object",
                            "properties": {
                                "full_name": {"type": "string"},
                                "headline": {"type": "string"},
                                "current_company": {"type": "string"},
                                "location": {"type": "string"},
                                "about": {"type": "string"},
                                "experience": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "company": {"type": "string"},
                                            "duration": {"type": "string"},
                                            "description": {"type": "string"}
                                        }
                                    }
                                },
                                "education": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "school": {"type": "string"},
                                            "degree": {"type": "string"},
                                            "field_of_study": {"type": "string"},
                                            "duration": {"type": "string"}
                                        }
                                    }
                                },
                                "skills": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            )
            
            if response and response.get('data'):
                return {
                    "data": {
                        "full_name": response['data'].get('full_name', ''),
                        "headline": response['data'].get('headline', ''),
                        "current_company": response['data'].get('current_company', ''),
                        "location": response['data'].get('location', ''),
                        "about": response['data'].get('about', ''),
                        "experience": response['data'].get('experience', []),
                        "education": response['data'].get('education', []),
                        "skills": response['data'].get('skills', [])
                    }
                }
        except Exception as e:
            logger.error(f"Firecrawl scraping failed for {linkedin_url}. Error: {e.__class__.__name__}: {str(e)}")
        return None

    async def _fetch_via_scrapingbee(self, linkedin_url: str) -> Optional[Dict]:
        """Scrape LinkedIn profile using ScrapingBee and LLM extraction as a fallback."""
        if not self.scrapingbee_api_key:
            return None

        logger.info(f"Attempting to scrape {linkedin_url} via ScrapingBee.")
        params = {
            'api_key': self.scrapingbee_api_key,
            'url': linkedin_url,
            'render_js': 'true',
            'premium_proxy': 'true', # Use residential proxies for better success
            'wait_for': 'main.scaffold-layout__main', # Wait for main content to load
            'forward_headers': 'true' # Tell ScrapingBee to use the headers we send it
        }
        request_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        }

        if self.linkedin_cookie:
            logger.info("Using LinkedIn session cookie for ScrapingBee.")
            params['cookies'] = f"li_at={self.linkedin_cookie}"

        try:
            async with self.session.get('https://app.scrapingbee.com/api/v1/', params=params, headers=request_headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ScrapingBee request failed with status {response.status}: {error_text[:200]}")
                    return None
                
                html_content = await response.text()
                # Convert HTML to clean text for the LLM
                text_content = html2text.html2text(html_content)
                
                # Now use an LLM to extract the data from the text
                extraction_prompt = f"""
                Analyze the following text content from a LinkedIn profile and extract the information into a JSON object.
                The text is messy, do your best to find the relevant information.

                Text Content:
                ---
                {text_content[:15000]}
                ---

                JSON Schema to populate:
                {{
                    "full_name": "string", "headline": "string", "current_company": "string",
                    "location": "string", "about": "string",
                    "experience": [{{ "title": "string", "company": "string", "duration": "string" }}],
                    "education": [{{ "school": "string", "degree": "string", "duration": "string" }}],
                    "skills": ["string"]
                }}

                Instructions:
                - For 'current_company', use the company from the most recent experience entry.
                - If a piece of information is not present, use a sensible default (e.g., empty string, empty list).
                - Only return the JSON object, with no other text or explanations.
                """
                
                response_str = await self.llm.query_llm(extraction_prompt)
                if not response_str: return None
                
                json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                return {"data": json.loads(json_match.group(0))} if json_match else None

        except Exception as e:
            logger.error(f"ScrapingBee processing failed for {linkedin_url}. Error: {e.__class__.__name__}: {str(e)}")
            return None

    def _normalize_rapidapi_data(self, data: Dict) -> Dict:
        """Normalize RapidAPI response to our standard format"""
        profile = data.get('profile', {})
        experiences = profile.get('experience', [])
        current_company = ""
        
        if experiences:
            current_exp = experiences[0]
            current_company = current_exp.get('company', {}).get('name', '') if isinstance(current_exp.get('company'), dict) else current_exp.get('company', '')
            
        return {
            "data": {
                "full_name": profile.get('full_name', ''),
                "headline": profile.get('headline', ''),
                "current_company": current_company,
                "location": profile.get('location', ''),
                "education": [{
                    "school": edu.get('school', {}).get('name', ''),
                    "degree": edu.get('degree', ''),
                    "field_of_study": edu.get('field_of_study', ''),
                    "duration": f"{edu.get('start_date', '')} - {edu.get('end_date', '')}"
                } for edu in profile.get('education', [])],
                "experience": [{
                    "title": exp.get('title', ''),
                    "company": exp.get('company', {}).get('name', '') if isinstance(exp.get('company'), dict) else exp.get('company', ''),
                    "duration": f"{exp.get('start_date', '')} - {exp.get('end_date', '')}",
                    "description": exp.get('description', '')
                } for exp in experiences],
                "skills": [skill.get('name', '') for skill in profile.get('skills', [])],
                "about": profile.get('summary', '')
            }
        }


    def _generate_mock_profile(self, linkedin_url: str, google_profile: Dict) -> Dict:
        """
        Generates a more realistic mock profile using data from the Google Search result as a fallback.
        """
        logger.info(f"Generating dynamic mock profile for {linkedin_url} from Google Search data.")
        
        # Initialize fields
        name, headline, company, location = "Not Available", "Not Available", "Not Available", "Not Available"
        
        # Use the more robust parsing logic
        google_title = google_profile.get('title', '')
        if google_title:
            cleaned_title = google_title.replace(' | LinkedIn', '').replace('...', '').strip()
            parts = [p.strip() for p in cleaned_title.split(' - ')]
            
            if parts:
                name = parts[0]
            
            if len(parts) > 1:
                last_part = parts[-1]
                if self._is_location(last_part):
                    location = last_part
                else:
                    company = last_part
                
                if len(parts) > 2:
                    headline = ' - '.join(parts[1:-1])
                elif len(parts) == 2:
                    # If it wasn't a location, it's likely the headline and company are the same
                    headline = parts[1]

        about = google_profile.get('snippet', 'No summary available.')

        return {
            "data": {
                "full_name": name,
                "headline": headline,
                "current_company": company,
                "location": location,
                "education": [],
                "experience": [
                    {"title": headline, "company": company, "duration": "Not Available"}
                ],
                "skills": [],
                "about": about
            }
        }

    def _parse_google_pagemap(self, google_profile: Dict) -> Dict:
        """Extracts structured data from Google's search result pagemap."""
        pagemap = google_profile.get("pagemap", {})
        person_data = pagemap.get("person", [{}])[0]
        hcard_data = pagemap.get("hcard", [{}])[0]

        return {
            "name": person_data.get("name") or hcard_data.get("fn"),
            "headline": person_data.get("role") or hcard_data.get("title"),
            "company": person_data.get("org"),
            "location": person_data.get("location"),
        }

    async def _fetch_candidate_data(self, linkedin_url: str, google_profile: Dict) -> Optional[Candidate]:
        """
        Fetches candidate data using a multi-layered fallback strategy.
        The primary logic is now consolidated within the _fetch_linkedin_profile and its fallbacks.
        This function now primarily orchestrates the call and constructs the final Candidate object.
        """
        # Layer 1: Attempt to get rich data from scrapers or the dynamic mock profile
        scraped_profile = await self._fetch_linkedin_profile(linkedin_url, google_profile)
        scraped_data = scraped_profile.get('data', {}) if scraped_profile else {}
        
        # Layer 2: Use Google Pagemap to fill any remaining gaps, as it can be more reliable.
        pagemap_data = self._parse_google_pagemap(google_profile)
        
        # Layer 3: Parse the Google title as a final fallback for core fields.
        google_title = google_profile.get('title', '')
        title_parts = [p.strip() for p in google_title.replace(' | LinkedIn', '').replace('...', '').split(' - ')]
        title_name, title_headline, title_company, title_location = None, None, None, None
        if title_parts:
            title_name = title_parts[0]
            if len(title_parts) > 1:
                last_part = title_parts[-1]
                if self._is_location(last_part):
                    title_location = last_part
                else:
                    title_company = last_part # Can also be a headline
                
                if len(title_parts) > 2:
                    title_headline = ' - '.join(title_parts[1:-1])
                elif not title_location: # if len is 2 and it's not a location, it's a headline
                    title_headline = title_parts[1]

        # Consolidate data, giving priority to scraped data, then pagemap, then title parse
        name = scraped_data.get('full_name') or pagemap_data.get('name') or title_name
        headline = scraped_data.get('headline') or pagemap_data.get('headline') or title_headline
        current_company = scraped_data.get('current_company') or pagemap_data.get('company') or title_company
        location = scraped_data.get('location') or pagemap_data.get('location') or title_location
        about = scraped_data.get('about') or google_profile.get('snippet')

        # Final cleanup for name
        if not name and google_title:
            name = google_title.split(' - ')[0].strip()

        return Candidate(
            name=(name.strip() if name else "Not Available"),
            linkedin_url=linkedin_url,
            headline=headline or "Not Available",
            current_company=current_company or "Not Available",
            location=location or "Not Available",
            education=scraped_data.get('education', []),
            experience=scraped_data.get('experience', []),
            skills=scraped_data.get('skills', []),
            about=about
        )

    def _is_location(self, text: str) -> bool:
        """Checks if a string is likely a location."""
        text_lower = text.lower()
        return (
            ',' in text or 
            'united states' in text_lower or 
            'area' in text_lower or
            any(city in text_lower for city in ['san francisco', 'new york', 'london', 'mountain view'])
        )

    async def score_candidates(self, candidates: List[Candidate], config: JobConfig) -> List[ScoredCandidate]:
        scored = []
        for candidate in candidates:
            # Calculate individual scores with more granularity
            education_score = self._calculate_education_score(candidate.education)
            experience_score = self._calculate_experience_score(candidate.experience, config.experience_years)
            skills_score = self._calculate_skills_score(candidate.skills, config.required_skills, config.nice_to_have_skills)
            company_score = self._calculate_company_score(candidate.current_company, config.company_name)
            location_score = self._calculate_location_score(candidate.location, config.location)
            tenure_score = self._calculate_tenure_score(candidate.experience)

            breakdown = ScoreBreakdown(
                education=round(education_score, 1),
                experience=round(experience_score, 1),
                skills=round(skills_score, 1),
                company_fit=round(company_score, 1),
                location=round(location_score, 1),
                tenure=round(tenure_score, 1)
            )

            # Weighted average with more emphasis on skills and experience
            total_score = (
                education_score * 0.15 +
                experience_score * 0.30 +
                skills_score * 0.30 +
                company_score * 0.10 +
                location_score * 0.10 +
                tenure_score * 0.05
            )

            scored.append(ScoredCandidate(
                candidate=candidate,
                fit_score=round(total_score, 1),
                score_breakdown=breakdown
            ))

        return sorted(scored, key=lambda x: x.fit_score, reverse=True)

    def _calculate_education_score(self, education: List[Dict[str, str]]) -> float:
        if not education:
            return 5.0  # Average score for no education info
        
        elite_schools = {
            "mit": 10.0, "stanford": 10.0, "harvard": 9.5, 
            "cmu": 9.5, "berkeley": 9.5, "oxford": 9.0,
            "cambridge": 9.0, "caltech": 9.5, "princeton": 9.0
        }
        
        top_tier = {
            "university of pennsylvania": 8.5, "columbia": 8.5,
            "university of chicago": 8.5, "cornell": 8.5,
            "university of michigan": 8.0, "ucla": 8.0,
            "university of texas": 7.5, "university of illinois": 7.5
        }
        
        max_score = 6.0  # Default for unknown schools
        
        for edu in education:
            school = edu.get("school", "").lower()
            degree = edu.get("degree", "").lower()
            
            # Check elite schools first
            for school_pattern, score in elite_schools.items():
                if school_pattern in school:
                    max_score = max(max_score, score)
                    break
            
            # Check top tier schools
            for school_pattern, score in top_tier.items():
                if school_pattern in school:
                    max_score = max(max_score, score)
                    break
            
            # Bonus for advanced degrees
            if "phd" in degree or "doctor" in degree:
                max_score = min(max_score + 1.0, 10.0)
            elif "master" in degree or "ms" in degree:
                max_score = min(max_score + 0.5, 10.0)
                
        return max_score

    def _parse_duration_to_years(self, duration_str: str) -> float:
        if not duration_str or not isinstance(duration_str, str):
            return 0.0

        duration_str = duration_str.lower()
        
        # Format: "X years Y months"
        match = re.search(r'(\d+)\s+years?.*(\d+)\s+mos?', duration_str)
        if match:
            return float(match.group(1)) + float(match.group(2)) / 12.0

        # Format: "X years" or "X.Y years"
        match = re.search(r'(\d+\.?\d*)\s+years?', duration_str)
        if match:
            return float(match.group(1))

        # Format: "Y months"
        match = re.search(r'(\d+)\s+mos?', duration_str)
        if match:
            return float(match.group(1)) / 12.0

        # Format: "YYYY - YYYY" or "Mon YYYY - Present"
        if '-' in duration_str:
            try:
                start_str, end_str = [s.strip() for s in duration_str.split('-')]
                start_year_match = re.search(r'(\d{4})', start_str)
                start_year = int(start_year_match.group(1)) if start_year_match else datetime.now().year
                if 'present' in end_str or 'now' in end_str:
                    end_year = datetime.now().year
                else:
                    end_year_match = re.search(r'(\d{4})', end_str)
                    end_year = int(end_year_match.group(1)) if end_year_match else start_year
                return max(0, end_year - start_year)
            except (ValueError, IndexError):
                return 0.0
        return 0.0

    def _calculate_experience_score(self, experience: List[Dict[str, str]], required_years: int) -> float:
        if not experience:
            return 4.0  # Below average for no experience info
        
        total_years = 0
        relevant_experience = 0
        
        for exp in experience:
            years = self._parse_duration_to_years(exp.get("duration", ""))
            total_years += years
            title = exp.get("title", "").lower()

        
        if total_years == 0:
            return 4.0
        
        # Calculate score based on required years and relevance
        years_score = min(total_years / required_years * 5.0, 5.0)  # Max 5 points for years
        relevance_score = (relevant_experience / total_years) * 5.0  # Max 5 points for relevance
        
        total = years_score + relevance_score
        
        # Cap at 10 and ensure minimum of 4
        return max(min(total, 10.0), 4.0)

    def _calculate_skills_score(self, skills: List[str], required_skills: List[str], nice_to_have_skills: List[str]) -> float:
        if not skills:
            return 4.0  # Below average for no skills info
        
        # Normalize all skills to lowercase
        candidate_skills = [s.lower() for s in skills]
        required_skills = [s.lower() for s in required_skills]
        nice_to_have_skills = [s.lower() for s in nice_to_have_skills]
        
        # Calculate matches
        required_matches = sum(1 for skill in required_skills if skill in candidate_skills)
        nice_matches = sum(1 for skill in nice_to_have_skills if skill in candidate_skills)
        
        # Base score based on required skills (70% of total)
        base_score = (required_matches / len(required_skills)) * 7.0 if required_skills else 0.0
        
        # Bonus for nice-to-have skills (30% of total)
        bonus_score = (nice_matches / len(nice_to_have_skills)) * 3.0 if nice_to_have_skills else 0.0
        
        total = base_score + bonus_score
        
        # Ensure score is between 4 and 10
        return max(min(total, 10.0), 4.0)

    def _calculate_company_score(self, current_company: str, target_company: str) -> float:
        if not current_company:
            return 6.0  # Neutral score for no company info
        
        # List of prestigious companies that get higher scores
        top_companies = [
            "google", "microsoft", "apple", "amazon", "meta", 
            "nvidia", "openai", "deepmind", "tesla", "netflix"
        ]
        
        current = current_company.lower()
        target = target_company.lower()
        
        # Heavily penalize if candidate already works at the target company (using word boundaries)
        if re.search(r'\b' + re.escape(target) + r'\b', current):
            return 1.0
            
        # Bonus if they work at a well-known top tech company
        if any(company in current for company in top_companies):
            return 8.5
            
        return 6.5  # Default for other companies

    def _calculate_location_score(self, candidate_location: str, target_location: str) -> float:
        if not candidate_location:
            return 6.0  # Neutral score for no location info
            
        candidate_loc = candidate_location.lower()
        target_loc = target_location.lower()
        
        # Exact match
        if target_loc in candidate_loc:
            return 10.0
            
        # Same city or region
        target_city = target_loc.split(",")[0].strip()
        if target_city in candidate_loc:
            return 9.0
            
        # Same state
        if len(target_loc.split(",")) > 1:
            target_state = target_loc.split(",")[1].strip()
            if target_state in candidate_loc:
                return 8.0
                
        # Same country
        if "united states" in candidate_loc and "united states" in target_loc:
            return 7.0
            
        return 6.0  # Default for no match

    def _calculate_tenure_score(self, experience: List[Dict[str, str]]) -> float:
        if not experience:
            return 5.0  # Average for no info
            
        tenures = []
        
        for exp in experience:
            duration = exp.get("duration", "")
            if "year" in duration:
                try:
                    if "-" in duration:  # Date range format
                        parts = duration.split("-")
                        if len(parts) >= 2:
                            start = parts[0].strip()
                            end = parts[1].strip()
                            
                            # Handle partial dates
                            if len(start) == 4:  # Just year
                                start_year = int(start)
                                if len(end) == 4:  # Just year
                                    end_year = int(end)
                                else:  # Month year or present
                                    end_year = datetime.now().year
                                years = end_year - start_year
                            else:  # Month year
                                start_year = int(start[-4:]) if len(start) >= 4 else datetime.now().year
                                if "present" in end.lower():
                                    end_year = datetime.now().year
                                else:
                                    end_year = int(end[-4:]) if len(end) >= 4 else datetime.now().year
                                years = end_year - start_year
                            
                            tenures.append(years)
                    else:  # "X years" format
                        years = float(duration.split()[0])
                        tenures.append(years)
                except (ValueError, IndexError):
                    continue
        
        if not tenures:
            return 5.0
            
        avg_tenure = sum(tenures) / len(tenures)
        
        if avg_tenure >= 4.0: return 9.5
        if avg_tenure >= 3.0: return 8.5
        if avg_tenure >= 2.0: return 7.5
        if avg_tenure >= 1.5: return 6.5
        return 5.0  # Default for short tenures

    async def generate_outreach(self, candidates: List[ScoredCandidate], config: JobConfig) -> List[Dict]:
        messages = []
        for candidate in candidates[:5]:  # Top 5 candidates
            prompt = f"""
            You are a senior tech recruiter writing a personalized, authentic, and concise LinkedIn outreach message (70-120 words).
            
            **Candidate to Contact:**
            - Name: {candidate.candidate.name}
            - Role & Company: {candidate.candidate.headline or 'a relevant role'} at {candidate.candidate.current_company or 'their current company'}
            - Location: {candidate.candidate.location or 'their current area'}
            
            **Job Opportunity:**
            - Company: {config.company_name}
            - Key Requirements: {', '.join(config.required_skills)}
            - Location: {config.location}
            
            **Instructions:**
            1. **Personalize the Hook:** Start with a hook that references the candidate's background.
               - **If Role & Company are available and sensible:** Use them. (e.g., "I came across your profile and was impressed by your work as a {candidate.candidate.headline}...").
               - **If Role or Company is missing or looks like a location:** Use a more general but still personal hook. (e.g., "I came across your profile and your background in {candidate.candidate.location or 'your area'} caught my eye..." or "Your experience with skills like {config.required_skills[0]} is very relevant...").
               - **CRITICAL:** Do NOT use placeholders like "[Current Role]" in the final message.
            2. **Bridge to Opportunity:** Briefly introduce the opportunity at {config.company_name} and connect it to their experience.
            3. **Call to Action:** End with a clear, low-pressure call to action.
            4. **Signature:** Sign off with "Best,\\n{config.company_name} Hiring Manager".

            **CRITICAL:** Your entire output must be ONLY the message text. Do NOT include a subject line, and absolutely no explanations, comments, or analysis about why you wrote the message the way you did.not even texts like here is your outreach message.Here is the personalized LinkedIn outreach message
            """
            
            response = await self.llm.query_llm(prompt)
            if not response:
                response = f"Hi {candidate.candidate.name.split()[0]},\n\nI came across your profile and was impressed by your background. We have an opportunity at {config.company_name} that seems like a strong match for your skills. Would you be open to a brief chat to learn more?\n\nBest,\n{config.company_name} Hiring Manager"
            
            messages.append({
                "name": candidate.candidate.name,
                "linkedin_url": candidate.candidate.linkedin_url,
                "message": response.strip()
            })
        return messages

    async def track_engagement(self, linkedin_url: str, status: str):
        if linkedin_url not in self.engagement_db:
            self.engagement_db[linkedin_url] = EngagementTracker()
        
        tracker = self.engagement_db[linkedin_url]
        tracker.last_contact = datetime.now()
        tracker.response_status = status
        
        if status == "contacted":
            asyncio.create_task(self.schedule_follow_up(linkedin_url))

    async def schedule_follow_up(self, linkedin_url: str):
        await asyncio.sleep(5 * 24 * 3600)  # 5 days
        
        if (linkedin_url in self.engagement_db and 
            self.engagement_db[linkedin_url].response_status == "contacted"):
            
            tracker = self.engagement_db[linkedin_url]
            tracker.follow_up_count += 1
            
            candidate = next((c for c in self.cache.values() 
                            if isinstance(c, Candidate) and c.linkedin_url == linkedin_url), None)
            
            if candidate:
                prompt = f"Generate polite follow-up message for {candidate.name} who didn't respond to initial outreach"
                follow_up = await self.llm.query_llm(prompt)
                logger.info(f"Follow-up ready for {candidate.name}: {follow_up[:50]}...")

    async def get_competitor_insights(self, skills: List[str]):
        if not skills:
            return {}
            
        query = f"site:linkedin.com/jobs ({' OR '.join(skills)})"
        jobs = await self._search_google(query)
        
        companies = {}
        for job in jobs:
            company = re.search(r"at (.*?) \(", job.get('title', ''))
            if company:
                companies[company.group(1)] = companies.get(company.group(1), 0) + 1
                
        self.competitor_insights = {
            "top_hiring_companies": sorted(companies.items(), key=lambda x: -x[1])[:5],
            "trending_skills": await self._identify_trending_skills(jobs)
        }
        return self.competitor_insights

    async def _identify_trending_skills(self, jobs: List[Dict]) -> List[str]:
        skills_counter = {}
        for job in jobs:
            if 'title' in job:
                for skill in self.config.required_skills + self.config.nice_to_have_skills:
                    if skill.lower() in job['title'].lower():
                        skills_counter[skill] = skills_counter.get(skill, 0) + 1
        return sorted(skills_counter.items(), key=lambda x: -x[1])[:3]

def identify_strengths(candidate: ScoredCandidate) -> List[str]:
    strengths = []
    if candidate.score_breakdown.skills >= 8:
        strengths.append("Excellent skills match")
    if candidate.score_breakdown.experience >= 8:
        strengths.append("Strong relevant experience")
    if candidate.score_breakdown.education >= 8:
        strengths.append("Top-tier education")
    return strengths or ["Good overall fit"]

def identify_concerns(candidate: ScoredCandidate) -> List[str]:
    concerns = []
    if candidate.score_breakdown.skills < 5:
        concerns.append("Limited required skills")
    if candidate.score_breakdown.experience < 5:
        concerns.append("Limited relevant experience")
    if candidate.score_breakdown.tenure < 5:
        concerns.append("Short job tenure history")
    return concerns or ["No significant concerns"]

def format_outreach_message(message: str) -> str:
    lines = [line.strip() for line in message.split('\n') if line.strip()]
    
    # Remove common conversational filler from the start
    if lines and (lines[0].lower().startswith("here's") or lines[0].lower().startswith("subject:") or lines[0].lower().startswith("hi ")):
        lines = lines[1:]
        
    # Re-add greeting if it was stripped
    cleaned_message = '\n'.join(lines).strip()
    if not cleaned_message.lower().startswith("hi"):
         return cleaned_message # Return as is if it doesn't look like a message body
    return cleaned_message

def calculate_data_quality_score(candidates: List[ScoredCandidate]) -> float:
    """Calculate a data quality score based on completeness"""
    if not candidates:
        return 0.0
    
    total_score = 0.0
    for candidate in candidates:
        completeness = 0
        if candidate.candidate.name: completeness += 1
        if candidate.candidate.headline: completeness += 1
        if candidate.candidate.current_company: completeness += 1
        if candidate.candidate.location: completeness += 1
        if candidate.candidate.about: completeness += 1
        if candidate.candidate.skills: completeness += 1
        
        total_score += (completeness / 6)  # Normalize to 0-1
        
    return round((total_score / len(candidates)) * 10, 1)  # Scale to 0-10

app = FastAPI(
    title="LinkedIn Sourcing API",
    description="API for sourcing and scoring LinkedIn candidates",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

agent = LinkedInSourcingAgent()

@app.on_event("startup")
async def startup_event():
    """Initialize the agent's session when the app starts."""
    await agent.initialize_session()
    await agent.initialize_driver()

@app.on_event("shutdown")
async def shutdown_event():
    """Close the agent's session when the app shuts down."""
    await agent.close_session()
    await agent.close_driver()

@app.post("/source_candidates", response_model=Dict)
async def source_candidates(job: JobDescription):
    """Source and score candidates based on job description"""
    if not agent.rapidapi_key and not agent.firecrawl_client.api_key and not agent.scrapingbee_api_key:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "API Keys Missing",
                "message": "No scraping API keys (e.g., RAPIDAPI_KEY, FIRECRAWL_API_KEY, SCRAPINGBEE_API_KEY) are configured in your .env file. The agent cannot fetch real data.",
                "request_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    try:
        candidates = await agent.search_linkedin(job.description, job.config)
        scored = await agent.score_candidates(candidates, job.config)
        messages = await agent.generate_outreach(scored, job.config)

        formatted_candidates = []
        for candidate in scored[:5]:
            message = next(
                (m['message'] for m in messages if m['linkedin_url'] == candidate.candidate.linkedin_url),
                "No message generated"
            )

            formatted_candidate = {
                "candidate_info": {
                    "name": candidate.candidate.name or "Not Available",
                    "profile_url": candidate.candidate.linkedin_url,
                    "current_position": candidate.candidate.headline or "Not Available",
                    "current_company": candidate.candidate.current_company or "Not Available",
                    "location": candidate.candidate.location or "Not Available",
                    "summary": (candidate.candidate.about[:250] + "...") if candidate.candidate.about else "Not Available"
                },
                "match_analysis": {
                    "overall_score": candidate.fit_score,
                    "score_breakdown": {
                        "education": candidate.score_breakdown.education,
                        "experience": candidate.score_breakdown.experience,
                        "skills": candidate.score_breakdown.skills,
                        "company_fit": candidate.score_breakdown.company_fit,
                        "location": candidate.score_breakdown.location,
                        "tenure": candidate.score_breakdown.tenure
                    },
                    "strengths": identify_strengths(candidate),
                    "potential_concerns": identify_concerns(candidate)
                },
                "outreach": {
                    "message": format_outreach_message(message),
                    "suggested_follow_up": "3-5 business days",
                    "personalization_tips": [
                        f"Mention their experience with {candidate.candidate.skills[0]}" if candidate.candidate.skills else "Highlight relevant skills",
                        "Reference the company mission if possible"
                    ]
                }
            }
            formatted_candidates.append(formatted_candidate)

        response = {                                                                        
            "metadata": {
                "job_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "search_parameters": job.config.dict(),
                "metrics": {
                    "total_candidates": len(scored),
                    "average_score": round(sum(s.fit_score for s in scored)/len(scored), 1) if scored else 0,
                    "top_candidate_score": scored[0].fit_score if scored else 0,
                    "data_quality": calculate_data_quality_score(scored)
                }
            },
            "results": formatted_candidates
        }

        return JSONResponse(
            content=response,
            headers={
                "X-API-Version": "1.1.0",
                "X-Request-ID": str(uuid4())
            }
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing error",
                "message": "We're having trouble processing your request. Please try again later.",
                "request_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/engagement_stats")
async def get_engagement_stats():
    contacted = sum(1 for t in agent.engagement_db.values() if t.response_status == "contacted")
    replied = sum(1 for t in agent.engagement_db.values() if t.response_status == "replied")
    
    return {
        "total_contacted": contacted,
        "response_rate": f"{(replied/max(1,contacted)*100):.1f}%",
        "pending_followups": sum(1 for t in agent.engagement_db.values() 
                               if t.response_status == "contacted" 
                               and (datetime.now() - t.last_contact).days >= 5)
    }

@app.get("/competitor_insights")
async def get_competitor_insights(skills: str):
    skills_list = [s.strip() for s in skills.split(",")]
    return await agent.get_competitor_insights(skills_list)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )