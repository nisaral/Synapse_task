AI-Powered LinkedIn Sourcing Agent
An intelligent, AI-driven recruitment tool that automates candidate sourcing from LinkedIn. By providing a job description, the agent searches for relevant candidates, scrapes their profiles, scores them based on job requirements, and generates personalized outreach messages. Built for resilience, it uses a multi-layered fallback system to ensure robust data collection and high-quality results.
Key Features

Automated Job Analysis: Extracts structured job requirements from raw job descriptions using LLMs.
Dynamic Profile Discovery: Searches LinkedIn profiles via Google Custom Search API with a fallback to DuckDuckGo.
Robust Profile Scraping:
Primary: Selenium + LLM for reliable, bot-resistant scraping.
Fallbacks: RapidAPI, Firecrawl, and ScrapingBee APIs.
Final Fallback: Generates mock profiles from search snippets if all else fails.


Intelligent Scoring: Evaluates candidates on education, experience, skills, company, location, and tenure using a weighted scoring system.
Personalized Outreach: Generates tailored, professional LinkedIn outreach messages for top candidates.
Market Insights: Analyzes competitor hiring trends and trending skills in the industry.
Stateful Architecture: Persists job and candidate data in a SQLite database, with caching for efficiency.
API-Driven: Built with FastAPI for scalable, asynchronous operations.

How It Works

Input: Submit a job description via a POST /source_candidates request.
Job Analysis: The LLM extracts structured requirements (skills, experience, location, etc.).
Profile Discovery:
Generates a search query (e.g., site:linkedin.com/in/ "Senior Machine Learning Engineer" "Python" "Mountain View").
Queries Google Custom Search API; falls back to DuckDuckGo if needed.


Profile Scraping: For each profile URL:
Attempts Selenium + LLM extraction.
Falls back to RapidAPI, Firecrawl, or ScrapingBee if Selenium fails.
Uses search snippets as a last resort for mock profiles.


Candidate Scoring: Scores candidates (1-10) across multiple criteria, weighted by job configuration.
Outreach Generation: Creates personalized outreach messages for top candidates.
Output: Returns a JSON response with candidate details, scores, and outreach messages.

Technology Stack

Backend: FastAPI, Uvicorn
AI/ML: Groq (Llama 3.1, Mixtral), OpenRouter APIs
Search: Google Custom Search API, DuckDuckGo
Scraping: Selenium (ChromeDriver), RapidAPI, Firecrawl, ScrapingBee
Libraries:
httpx, aiohttp: Async HTTP requests
pydantic: Data validation
aiosqlite: Database management
beautifulsoup4: HTML parsing
PyPDF2: PDF processing
cachetools: Caching
python-dotenv: Environment variable management
html2text: HTML-to-text conversion



Setup and Installation
Prerequisites

Python 3.9+
Google Chrome browser
ChromeDriver (compatible with installed Chrome version)
Git

1. Clone the Repository
git clone <your-repository-url>
cd linkedin-sourcing-agent

2. Create a Virtual Environment
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

requirements.txt:
fastapi==0.115.0
uvicorn==0.30.6
httpx==0.27.2
aiohttp==3.10.5
pydantic==2.9.2
PyPDF2==3.0.1
aiosqlite==0.20.0
selenium==4.25.0
beautifulsoup4==4.12.3
cachetools==5.5.0
python-dotenv==1.0.1
html2text==2024.2.26
webdriver-manager==4.0.2

4. Install ChromeDriver

Windows:
Download ChromeDriver from chromedriver.chromium.org.
Add to PATH or place in project directory.


macOS:brew install chromedriver


Ubuntu:sudo apt update
sudo apt install chromium-chromedriver



Verify ChromeDriver:
chromedriver --version

5. Configure Environment Variables
Create a .env file in the project root with the following:
# LLM APIs (at least one required)
GROQ_API_KEY="gsk_..."
OPENROUTER_API_KEY="sk-or-..."

# LinkedIn Credentials (required for Selenium)
LINKEDIN_EMAIL="your-linkedin-email@example.com"
LINKEDIN_PASSWORD="your-linkedin-password"

# Google Custom Search (recommended)
GOOGLE_SEARCH_API_KEY="AIza..."
GOOGLE_SEARCH_CX="your_custom_search_engine_id"

# Scraper APIs (optional, recommended for resilience)
RAPIDAPI_KEY="your_rapidapi_key"
FIRECRAWL_API_KEY="fc-..."
SCRAPINGBEE_API_KEY="your_scrapingbee_key"

# LinkedIn Session Cookie (optional, for Firecrawl/ScrapingBee)
# Find in browser: DevTools -> Application -> Cookies -> linkedin.com -> li_at
LINKEDIN_LI_AT_COOKIE="your_li_at_cookie_value"


Google CSE: Create at programmablesearchengine.google.com with site:linkedin.com/in/*. Get API key from Google Cloud Console.
RapidAPI: Subscribe to LinkedIn Profile API at rapidapi.com.
Firecrawl/ScrapingBee: Optional; get keys from respective platforms.
LinkedIn Cookie: Open LinkedIn in Chrome, go to DevTools -> Application -> Cookies, copy li_at value.

6. Run the Application
uvicorn main:app --host 0.0.0.0 --port 8080

Access:

API: http://127.0.0.1:8080
Docs: http://127.0.0.1:8080/docs

API Usage
Source Candidates
Endpoint: POST /source_candidatesDescription: Finds, scores, and generates outreach for candidates.Body:
{
  "description": "Senior Machine Learning Engineer at Windsurf, building AI-powered developer tools. Requires 5+ years of experience in Python, LLMs, and machine learning. Based in Mountain View, $140-300k + equity.",
  "config": {
    "industry": "ai",
    "role_level": "senior",
    "location_preference": "mountain_view",
    "scoring_weights": {
      "education": 0.15,
      "experience": 0.30,
      "skills": 0.25,
      "company": 0.15,
      "location": 0.10,
      "tenure": 0.05
    },
    "search_limits": {
      "max_candidates": 10,
      "min_score": 6.0
    }
  }
}

Example curl:
curl -X POST "http://127.0.0.1:8080/source_candidates" \
-H "Content-Type: application/json" \
-d '{"description": "Senior Machine Learning Engineer at Windsurf, building AI-powered developer tools. Requires 5+ years of experience in Python, LLMs, and machine learning. Based in Mountain View, $140-300k + equity.", "config": {"industry": "ai", "role_level": "senior", "location_preference": "mountain_view", "scoring_weights": {"education": 0.15, "experience": 0.30, "skills": 0.25, "company": 0.15, "location": 0.10, "tenure": 0.05}, "search_limits": {"max_candidates": 10, "min_score": 6.0}}}'

Response:
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "candidates": [
    {
      "name": "John Doe",
      "linkedin_url": "https://linkedin.com/in/john-doe-123",
      "fit_score": 8.3,
      "score_breakdown": {
        "education": 9.0,
        "trajectory": 8.0,
        "company": 9.0,
        "skills": 8.5,
        "location": 10.0,
        "tenure": 7.0,
        "confidence": 8.5
      },
      "outreach_message": "Dear John Doe, Your expertise in Python and LLMs at Google aligns perfectly with our Senior Machine Learning Engineer role at Windsurf in Mountain View. We're innovating AI-powered developer tools. Can we schedule a call to discuss this opportunity?"
    }
  ]
}

Analyze Job Description
Endpoint: POST /analyze_job_descriptionDescription: Generates a JobConfig from a raw job description.Body:
{
  "description": "Job Title: Machine Learning Engineer. Location: Mountain View, CA. We need someone with 5-7 years of experience in Python and LLMs."
}

Example curl:
curl -X POST "http://127.0.0.1:8000/analyze_job_description" \
-H "Content-Type: application/json" \
-d '{"description": "Job Title: Machine Learning Engineer. Location: Mountain View, CA. We need someone with 5-7 years of experience in Python and LLMs."}'

Response:
{
  "required_skills": ["Python", "LLMs"],
  "nice_to_have_skills": [],
  "experience_years": 5,
  "education": ["Bachelor's in Computer Science"],
  "location": "Mountain View, CA",
  "industry_keywords": ["ai"]
}

Job Status
Endpoint: GET /job_status/{job_id}Description: Retrieves job details and candidate results.Example curl:
curl -X GET "http://127.0.0.1:8080/job_status/123e4567-e89b-12d3-a456-426614174000"

Rescore Candidates
Endpoint: POST /rescoreDescription: Rescores candidates for a job with updated weights.Body:
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "config": {
    "industry": "ai",
    "role_level": "senior",
    "location_preference": "mountain_view",
    "scoring_weights": {
      "education": 0.10,
      "experience": 0.25,
      "skills": 0.35,
      "company": 0.15,
      "location": 0.10,
      "tenure": 0.05
    },
    "search_limits": {
      "max_candidates": 10,
      "min_score": 6.0
    }
  }
}

Analytics
Endpoint: GET /analyticsDescription: Returns job processing stats.Example curl:
curl -X GET "http://127.0.0.1:8080/analytics"

Response:
{
  "jobs_processed": 1,
  "average_score": 8.5,
  "api_usage": {}
}

Deploying on Render

Prepare Files:

Ensure requirements.txt is up-to-date.
Create build.sh:#!/usr/bin/env bash
set -o errexit
apt-get update
apt-get install -y wget gnupg
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
apt-get update
apt-get install -y google-chrome-stable
pip install -r requirements.txt




Deploy:

Push code to a GitHub repository.
In Render, create a Web Service and connect your repository.
Settings:
Environment: Python
Build Command: bash build.sh
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT


Add .env variables in Render’s Environment tab.
Click Create Web Service.






Troubleshooting

API Errors:

Google CSE 403: Verify GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX. Test:curl "https://www.googleapis.com/customsearch/v1?key=$GOOGLE_SEARCH_API_KEY&cx=$GOOGLE_SEARCH_CX&q=site:linkedin.com/in/%20software%20engineer%20python%20mountain%20view&num=10"


RapidAPI 403: Ensure RAPIDAPI_KEY is active. Test:curl -X GET "https://linkedin-api8.p.rapidapi.com/get-profile-by-url?url=https%3A%2F%2Flinkedin.com%2Fin%2Fjohn-doe" -H "X-RapidAPI-Key: $RAPIDAPI_KEY" -H "X-RapidAPI-Host: linkedin-api8.p.rapidapi.com"


Groq/OpenRouter Failure: Check API keys and rate limits. Test:curl -X POST "https://api.groq.com/openai/v1/chat/completions" -H "Authorization: Bearer $GROQ_API_KEY" -H "Content-Type: application/json" -d '{"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "Hello"}]}'




Selenium Errors:

NewConnectionError: Ensure ChromeDriver is running:chromedriver --port=61989


Test Selenium:from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
driver.get("https://linkedin.com")
print(driver.title)
driver.quit()




Database Issues:

Verify sourcing.db:sqlite3 sourcing.db "SELECT * FROM jobs;"




Empty Candidate List:

Check logs for API/scraping failures.
Ensure LinkedIn credentials or LINKEDIN_LI_AT_COOKIE are valid.
Test with a known LinkedIn URL in POST /source_candidates.



Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
MIT License. See LICENSE for details.
Contact
For issues or feature requests, open a GitHub issue or contact your-email@example.com.
