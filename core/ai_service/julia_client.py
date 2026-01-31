
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class JuliaClient:
    """
    Client for interacting with the Julia POMDP Visualization Server.
    Integrates the belief state and policy decision into the Python AI workflow.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session_map: Dict[str, str] = {}  # Map student_id -> julia_session_id

    async def get_session(self, student_id: str) -> str:
        """Get or create a Julia session for the student."""
        if student_id in self.session_map:
            return self.session_map[student_id]
        
        # Create new session
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"studentId": student_id}
                async with session.post(f"{self.base_url}/session/start", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.session_map[student_id] = data["sessionId"]
                        logger.info(f"Created Julia session for {student_id}: {data['sessionId']}")
                        return data["sessionId"]
                    else:
                        logger.error(f"Failed to create Julia session: {await resp.text()}")
                        return ""
        except Exception as e:
            logger.error(f"Error connecting to Julia backend: {e}")
            return ""

    async def step(self, student_id: str, correctness: float, confidence: float, response_time: float) -> Dict[str, Any]:
        """
        Send interaction data to Julia and get the next optimal pedagogical action.
        """
        # Ensure session exists
        if student_id not in self.session_map:
            await self.get_session(student_id)

        observation = {
            "correctness": correctness,
            "confidence": confidence,
            "response_time": response_time
        }
        
        payload = {
            "studentId": student_id,
            "interaction": observation
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/session/step", json=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.error(f"Failed to step Julia model: {await resp.text()}")
                        return {}
        except Exception as e:
            logger.error(f"Error stepping Julia model: {e}")
            return {}

    async def get_diagnostics(self, student_id: str) -> Dict[str, Any]:
        """Retrieve diagnostic info (belief drift, uncertainty) for the student."""
        session_id = await self.get_session(student_id)
        if not session_id:
            return {}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/session/{session_id}/diagnostics") as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {}
        except Exception as e:
            logger.error(f"Error fetching diagnostics: {e}")
            return {}
