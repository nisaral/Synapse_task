from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
import asyncio
import logging
from cachetools import TTLCache
import re
import json
import httpx
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
@dataclass 
class EngagementTracker:
    last_contact: datetime = None
    response_status: str = "not_contacted"  # not_contacted/contacted/replied
    follow_up_count: int = 0

class LinkedInSourcingAgent:
    def __init__(self):
        # [Previous init code...]
        self.engagement_db = TTLCache(maxsize=500, ttl=timedelta(days=30))  # Tracks candidate engagement
        self.competitor_insights = {}  # Stores competitor hiring data

    # [All previous methods remain the same...]

    async def track_engagement(self, linkedin_url: str, status: str):
        """Track candidate engagement status"""
        if linkedin_url not in self.engagement_db:
            self.engagement_db[linkedin_url] = EngagementTracker()
        
        tracker = self.engagement_db[linkedin_url]
        tracker.last_contact = datetime.now()
        tracker.response_status = status
        
        if status == "contacted":
            # Schedule follow-up in 5 days
            asyncio.create_task(self.schedule_follow_up(linkedin_url))

    async def schedule_follow_up(self, linkedin_url: str):
        """Auto-schedule follow-up messages"""
        await asyncio.sleep(5 * 24 * 3600)  # 5 days
        
        if (linkedin_url in self.engagement_db and 
            self.engagement_db[linkedin_url].response_status == "contacted"):
            
            tracker = self.engagement_db[linkedin_url]
            tracker.follow_up_count += 1
            
            # Generate follow-up message
            candidate = next((c for c in self.cache.values() 
                            if isinstance(c, Candidate) and c.linkedin_url == linkedin_url), None)
            
            if candidate:
                prompt = f"Generate polite follow-up message for {candidate.name} who didn't respond to initial outreach"
                follow_up = await self.llm.query_llm(prompt)
                
                logger.info(f"Follow-up ready for {candidate.name}: {follow_up[:50]}...")
                # Here you'd integrate with your email/LinkedIn API

    async def get_competitor_insights(self, skills: List[str]):
        """Identify companies hiring similar profiles"""
        if not skills:
            return {}
            
        query = f"site:linkedin.com/jobs ({' OR '.join(skills)})"
        jobs = await self._search_google(query)
        
        # Simple competitor analysis
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
        """Extract trending skills from job postings"""
        skills_counter = {}
        for job in jobs:
            if 'title' in job:
                for skill in self.config.required_skills + self.config.nice_to_have_skills:
                    if skill.lower() in job['title'].lower():
                        skills_counter[skill] = skills_counter.get(skill, 0) + 1
        return sorted(skills_counter.items(), key=lambda x: -x[1])[:3]
@dataclass
class Candidate:
    name: str
    linkedin_url: str
    headline: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    education: List[Dict[str, str]] = None
    experience: List[Dict[str, str]] = None
    skills: List[str] = None

@dataclass
class ScoreBreakdown:
    education: float
    experience: float
    skills: float
    company: float
    location: float
    tenure: float

@dataclass
class ScoredCandidate:
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
        self.request_counts = {k: [] for k in self.rate_limits}

    async def query_llm(self, prompt: str, model_index: int = 0) -> Optional[str]:
        provider, model, url = self.models[model_index]
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY') if provider == 'groq' else os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        now = datetime.now()
        self.request_counts[provider] = [t for t in self.request_counts[provider] if (now - t) < timedelta(seconds=self.rate_limits[provider]["window"])]
        if len(self.request_counts[provider]) >= self.rate_limits[provider]["requests"]:
            wait_time = self.rate_limits[provider]["window"] - (now - self.request_counts[provider][0]).seconds
            await asyncio.sleep(wait_time)
        
        self.request_counts[provider].append(now)
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            response = await self.client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"API request failed for {provider}: {str(e)}")
            if model_index < len(self.models) - 1:
                return await self.query_llm(prompt, model_index + 1)
            return None

class LinkedInSourcingAgent:
    def __init__(self):
        self.llm = LLMService()
        self.cache = TTLCache(maxsize=100, ttl=3600)
        self.session = None
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY')
        self.rapidapi_host = "fresh-linkedin-profile-data.p.rapidapi.com"
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.google_cx = os.getenv('GOOGLE_SEARCH_CX')

    async def initialize_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def search_linkedin(self, job_description: str, config: JobConfig) -> List[Candidate]:
        cache_key = f"search_{hash(job_description)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        await self.initialize_session()
        search_query = await self._generate_search_query(config)
        logger.info(f"Searching with query: {search_query}")
        
        profiles = await self._search_google(search_query)
        candidates = []
        
        for profile in profiles[:10]:  # Limit to first 10 profiles
            linkedin_url = profile.get('link', '')
            if not linkedin_url.startswith('https://www.linkedin.com/in/'):
                continue
            candidate = await self._fetch_candidate_data(linkedin_url, profile)
            if candidate:
                candidates.append(candidate)

        self.cache[cache_key] = candidates
        return candidates

    async def _generate_search_query(self, config: JobConfig) -> str:
        # Simplified query generation without LLM
        base_query = f'site:linkedin.com/in/ ("{config.company_name}" OR "{config.location}")'
        skills_query = ' OR '.join(f'"{skill}"' for skill in config.required_skills)
        return f"{base_query} ({skills_query})"

    async def _search_google(self, query: str) -> List[Dict]:
        if not self.google_api_key or not self.google_cx:
            logger.warning("Google API credentials not configured - using mock data")
            return [
                {"link": "https://www.linkedin.com/in/johndoe", "title": "John Doe | LinkedIn"},
                {"link": "https://www.linkedin.com/in/janesmith", "title": "Jane Smith | LinkedIn"}
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

    async def _fetch_linkedin_profile(self, linkedin_url: str) -> Optional[Dict]:
        if not self.rapidapi_key:
            logger.warning("RapidAPI key not configured - using mock data")
            return {
                "data": {
                    "full_name": linkedin_url.split('/in/')[1].title(),
                    "headline": "Senior Software Engineer",
                    "company": {"name": "Windsurf"},
                    "location": "Mountain View, California",
                    "education": [{"school": "Stanford University", "degree": "MS", "field_of_study": "Computer Science"}],
                    "experience": [
                        {"title": "Senior Software Engineer", "company": "Windsurf", "duration": "3 years"},
                        {"title": "Software Engineer", "company": "Google", "duration": "4 years"}
                    ],
                    "skills": ["Python", "LLMs", "Machine Learning"]
                }
            }

        url = "https://fresh-linkedin-profile-data.p.rapidapi.com/get-profile-pdf-cv"
        params = {"linkedin_url": linkedin_url}
        headers = {
            "x-rapidapi-key": self.rapidapi_key,
            "x-rapidapi-host": self.rapidapi_host
        }

        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                logger.error(f"RapidAPI request failed with status: {response.status}")
                return None
        except Exception as e:
            logger.error(f"RapidAPI request failed: {str(e)}")
            return None

    async def _fetch_candidate_data(self, linkedin_url: str, google_profile: Dict) -> Optional[Candidate]:
        profile_data = await self._fetch_linkedin_profile(linkedin_url)
        if not profile_data:
            return None

        data = profile_data.get('data', {})
        return Candidate(
            name=data.get('full_name', google_profile.get('title', 'Unknown').replace(' | LinkedIn', '')),
            linkedin_url=linkedin_url,
            headline=data.get('headline', ''),
            current_company=data.get('company', {}).get('name', '') if data.get('company') else '',
            location=data.get('location', ''),
            education=[
                {
                    "school": edu.get('school', ''),
                    "degree": edu.get('degree', ''),
                    "field_of_study": edu.get('field_of_study', '')
                } for edu in data.get('education', [])
            ],
            experience=[
                {
                    "title": exp.get('title', ''),
                    "company": exp.get('company', ''),
                    "duration": exp.get('duration', '')
                } for exp in data.get('experience', [])
            ],
            skills=data.get('skills', [])
        )

    async def score_candidates(self, candidates: List[Candidate], config: JobConfig) -> List[ScoredCandidate]:
        scored = []
        for candidate in candidates:
            education_score = self._calculate_education_score(candidate.education)
            experience_score = self._calculate_experience_score(candidate.experience, config.experience_years)
            skills_score = await self._calculate_skills_score(candidate.skills, config)
            company_score = self._calculate_company_score(candidate.current_company, config.company_name)
            location_score = self._calculate_location_score(candidate.location, config.location)
            tenure_score = self._calculate_tenure_score(candidate.experience)

            breakdown = ScoreBreakdown(
                education=education_score,
                experience=experience_score,
                skills=skills_score,
                company=company_score,
                location=location_score,
                tenure=tenure_score
            )

            total_score = (
                education_score * 0.20 +
                experience_score * 0.25 +
                skills_score * 0.25 +
                company_score * 0.15 +
                location_score * 0.10 +
                tenure_score * 0.05
            )

            scored.append(ScoredCandidate(
                candidate=candidate,
                fit_score=total_score,
                score_breakdown=breakdown
            ))

        return sorted(scored, key=lambda x: x.fit_score, reverse=True)

    def _calculate_education_score(self, education: List[Dict[str, str]]) -> float:
        elite_schools = ["MIT", "Stanford", "Harvard", "CMU", "Berkeley"]
        if not education:
            return 5.0
        
        max_score = 0
        for edu in education:
            school = edu.get("school", "").lower()
            if any(elite.lower() in school for elite in elite_schools):
                max_score = max(max_score, 9.0)
            else:
                max_score = max(max_score, 6.0)
        return max_score

    def _calculate_experience_score(self, experience: List[Dict[str, str]], required_years: int) -> float:
        if not experience:
            return 5.0
        
        total_years = 0
        for exp in experience:
            duration = exp.get("duration", "")
            if "year" in duration:
                try:
                    years = float(duration.split()[0])
                    total_years += years
                except (ValueError, IndexError):
                    continue
        
        if total_years >= required_years + 2:
            return 10.0
        elif total_years >= required_years:
            return 8.0
        return 5.0

    async def _calculate_skills_score(self, skills: List[str], config: JobConfig) -> float:
        if not skills:
            return 5.0

        required_matches = sum(1 for skill in config.required_skills if skill.lower() in [s.lower() for s in skills])
        nice_matches = sum(1 for skill in config.nice_to_have_skills if skill.lower() in [s.lower() for s in skills])
        
        base_score = min(required_matches / len(config.required_skills) * 10, 10.0)
        bonus = nice_matches * 0.5
        return min(base_score + bonus, 10.0)

    def _calculate_company_score(self, current_company: str, target_company: str) -> float:
        if not current_company:
            return 5.0
        return 9.0 if target_company.lower() in current_company.lower() else 6.0

    def _calculate_location_score(self, candidate_location: str, target_location: str) -> float:
        if not candidate_location:
            return 6.0
        return 10.0 if target_location.lower() in candidate_location.lower() else 6.0

    def _calculate_tenure_score(self, experience: List[Dict[str, str]]) -> float:
        if not experience:
            return 5.0
        
        avg_tenure = sum(
            float(exp.get("duration", "0").split()[0]) 
            for exp in experience 
            if "year" in exp.get("duration", "")
        ) / len(experience)
        
        if avg_tenure >= 2.5: return 9.0
        if avg_tenure >= 2.0: return 8.0
        if avg_tenure >= 1.5: return 7.0
        return 5.0

    async def generate_outreach(self, candidates: List[ScoredCandidate], config: JobConfig) -> List[Dict]:
        messages = []
        for candidate in candidates[:5]:  # Top 5 candidates
            prompt = f"""Generate a LinkedIn outreach message for:
            Name: {candidate.candidate.name}
            Current Role: {candidate.candidate.headline}
            Company: {candidate.candidate.current_company}
            Skills: {', '.join(candidate.candidate.skills)}
            
            Target Company: {config.company_name}
            Role Requirements: {', '.join(config.required_skills)}
            Salary: {config.salary_range}
            
            Keep it professional (50-100 words) with call-to-action"""
            
            response = await self.llm.query_llm(prompt)
            messages.append({
                "name": candidate.candidate.name,
                "linkedin_url": candidate.candidate.linkedin_url,
                "message": response.strip() if response else "Could not generate message"
            })
        return messages

app = FastAPI()

@app.post("/source_candidates")
async def source_candidates(job: JobDescription):
    try:
        agent = LinkedInSourcingAgent()
        try:
            candidates = await agent.search_linkedin(job.description, job.config)
            scored = await agent.score_candidates(candidates, job.config)
            messages = await agent.generate_outreach(scored, job.config)
            
            return {
                "job_id": str(uuid4()),
                "candidates_found": len(scored),
                "top_candidates": [{
                    "name": s.candidate.name,
                    "linkedin_url": s.candidate.linkedin_url,
                    "fit_score": round(s.fit_score, 1),
                    "score_breakdown": {
                        "education": round(s.score_breakdown.education, 1),
                        "experience": round(s.score_breakdown.experience, 1),
                        "skills": round(s.score_breakdown.skills, 1),
                        "company": round(s.score_breakdown.company, 1),
                        "location": round(s.score_breakdown.location, 1),
                        "tenure": round(s.score_breakdown.tenure, 1)
                    },
                    "outreach_message": next(
                        (m['message'] for m in messages if m['linkedin_url'] == s.candidate.linkedin_url), 
                        "No message"
                    )
                } for s in scored[:5]]  # Return top 5 candidates
            }
        finally:
            await agent.close_session()
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/engagement_stats")
async def get_engagement_stats():
    agent = LinkedInSourcingAgent()
    try:
        contacted = sum(1 for t in agent.engagement_db.values() if t.response_status == "contacted")
        replied = sum(1 for t in agent.engagement_db.values() if t.response_status == "replied")
        
        return {
            "total_contacted": contacted,
            "response_rate": f"{(replied/max(1,contacted)*100):.1f}%",
            "pending_followups": sum(1 for t in agent.engagement_db.values() 
                                   if t.response_status == "contacted" 
                                   and (datetime.now() - t.last_contact).days >= 5)
        }
    finally:
        await agent.close_session()

@app.get("/competitor_insights")
async def get_competitor_insights(skills: str):
    agent = LinkedInSourcingAgent()
    try:
        skills_list = [s.strip() for s in skills.split(",")]
        return await agent.get_competitor_insights(skills_list)
    finally:
        await agent.close_session()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)