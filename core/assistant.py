import time
import requests
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from config import settings
from core.event_engine import EventEngine
from core.rag_memory import SurveillanceMemoryManager

class InvestigationAssistant:
    def __init__(self, rag_memory: SurveillanceMemoryManager, event_engine: EventEngine):
        self.rag_memory = rag_memory
        self.event_engine = event_engine
        self.api_key = settings.GEMINI_API_KEY

    def generate_response(self, user_query: str, history: Optional[List[Dict[str, str]]] = None,
                          operator_username: str = "Unknown") -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Processes a natural language query:
        1. Queries RAG memory to retrieve context.
        2. Routes to Gemini API (if key is set) or runs the Offline Summarizer fallback.
        3. Logs the query execution to the database audit trail.
        """
        start_time = time.time()
        
        # 1. Retrieve RAG memory documents
        sources = self.rag_memory.search_memory(user_query, k=5)
        retrieved_ids = [s["memory_id"] for s in sources]
        
        backend_used = "Offline"
        response_text = ""
        
        # 2. Try Gemini API if key is configured
        if self.api_key:
            try:
                response_text = self._call_gemini_api(user_query, sources, history or [])
                backend_used = "Gemini"
            except Exception as e:
                print(f"[RAG WARNING] Gemini API call failed: {e}. Falling back to Offline backend.")
                backend_used = "Offline"
                
        # 3. Fallback to First-Principles Deterministic Summarizer
        if backend_used == "Offline":
            response_text = self._generate_offline_summary(user_query, sources)
            
        latency_ms = (time.time() - start_time) * 1000
        
        # 4. Log to Audit Database
        self.event_engine.log_assistant_query(
            user_username=operator_username,
            query=user_query,
            retrieved_memory_ids=retrieved_ids,
            latency_ms=latency_ms,
            backend_used=backend_used
        )
        
        return response_text, sources, backend_used

    def _call_gemini_api(self, query: str, context_sources: List[Dict[str, Any]],
                        history: List[Dict[str, str]]) -> str:
        """Calls Google Gemini API (gemini-1.5-flash) via direct HTTP POST requests."""
        # Construct system prompt context
        context_str = "\n".join(
            f"- [{s['timestamp']}] [{s['entity_type'].upper()}] {s['document_text']}"
            for s in context_sources
        )
        
        system_instruction = (
            "You are VisionGuard's Natural Language Investigation Assistant. "
            "Your task is to answer the investigator's query based ONLY on the following retrieved surveillance memories. "
            "If the memories do not contain the answer, say 'I could not find any relevant sighting records in the surveillance database.' "
            "Format your answer as a structured markdown response. Highlight key timestamps, cameras, and target names.\n\n"
            f"Retrieved Surveillance Memories:\n{context_str}"
        )
        
        # Map conversation history to Gemini API contents format:
        # role can be "user" or "model" (not "assistant")
        contents = []
        for h in history:
            role = "model" if h.get("role") in ["assistant", "model"] else "user"
            contents.append({
                "role": role,
                "parts": [{"text": h.get("content", "")}]
            })
            
        # Append current query
        contents.append({
            "role": "user",
            "parts": [{"text": f"System Instruction: {system_instruction}\n\nUser Query: {query}"}]
        })
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": contents}
        
        response = requests.post(url, headers=headers, json=payload, timeout=8)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API returned status code {response.status_code}: {response.text}")
            
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse Gemini API response: {data}") from e

    def _generate_offline_summary(self, query: str, context_sources: List[Dict[str, Any]]) -> str:
        """Generates a structured chronological investigation report from retrieved memories."""
        if not context_sources:
            return (
                "### Executive Summary\n"
                "I could not find any relevant sighting records in the surveillance database."
            )

        # 1. Executive Summary details
        total_sources = len(context_sources)
        entity_counts = {"event": 0, "alert": 0, "tracklet": 0, "movement": 0}
        unique_cams = set()
        unique_pids = set()
        unique_alerts = set()
        unique_evidence = set()
        sim_scores = []
        
        for s in context_sources:
            t = s["entity_type"]
            if t in entity_counts:
                entity_counts[t] += 1
            if s["camera_id"]:
                unique_cams.add(s["camera_id"])
            if s["person_id"]:
                unique_pids.add(s["person_id"])
            if t == "alert":
                unique_alerts.add(s["reference_id"])
            if s["evidence_path"]:
                unique_evidence.add(s["evidence_path"])
            sim_scores.append(s["similarity"])
            
        avg_sim = sum(sim_scores) / total_sources if total_sources > 0 else 0.0
        
        summary_parts = []
        if entity_counts["event"] > 0:
            summary_parts.append(f"{entity_counts['event']} sighting event(s)")
        if entity_counts["alert"] > 0:
            summary_parts.append(f"{entity_counts['alert']} alert(s)")
        if entity_counts["tracklet"] > 0:
            summary_parts.append(f"{entity_counts['tracklet']} tracklet(s)")
        if entity_counts["movement"] > 0:
            summary_parts.append(f"{entity_counts['movement']} movement transition(s)")
            
        summary_str = ", ".join(summary_parts)
        
        # 2. Reconstruct chronological timeline
        sorted_sources = sorted(context_sources, key=lambda x: x["timestamp"])
        timeline_lines = []
        for idx, s in enumerate(sorted_sources):
            # Clean timestamp for readability
            t_str = s["timestamp"]
            try:
                dt = datetime.fromisoformat(s["timestamp"].replace("Z", ""))
                t_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            
            # Map type representation
            emoji = {
                "event": "[Sighting]",
                "alert": "[Alert]",
                "tracklet": "[Tracklet]",
                "movement": "[Movement]"
            }.get(s["entity_type"], "[Memory]")
            
            timeline_lines.append(f"{idx+1}. **{t_str}** - {emoji} {s['document_text']}")

        # 3. Retrieve camera locations and person names from database
        camera_details = []
        for cid in sorted(list(unique_cams)):
            loc = "Unknown Location"
            try:
                with self.event_engine._get_connection() as conn:
                    cursor = conn.execute("SELECT location FROM cameras WHERE camera_id = ?", (cid,))
                    row = cursor.fetchone()
                    if row:
                        loc = row[0]
            except Exception:
                pass
            camera_details.append(f"- **{cid}** ({loc})")
            
        person_details = []
        for pid in sorted(list(unique_pids)):
            name = "Unknown Target"
            try:
                p_info = self.event_engine.get_person(pid)
                if p_info:
                    name = p_info.get("name", "Unknown Target")
            except Exception:
                pass
            person_details.append(f"- **{pid}** ({name})")

        # 4. Format report sections
        report = []
        report.append("### Executive Summary")
        report.append(f"The surveillance database contains **{total_sources}** records matching your query. This includes: {summary_str}.")
        report.append("")
        
        report.append("### Chronological Timeline")
        report.extend(timeline_lines)
        report.append("")
        
        report.append("### Cameras Involved")
        if camera_details:
            report.extend(camera_details)
        else:
            report.append("- No camera metadata recorded.")
        report.append("")
        
        report.append("### Persons Involved")
        if person_details:
            report.extend(person_details)
        else:
            report.append("- No person identity matches recorded.")
        report.append("")
        
        report.append("### Alerts Triggered")
        if unique_alerts:
            for aid in sorted(list(unique_alerts)):
                # Query alert status if available
                status = "ACTIVE"
                try:
                    alert_info = self.event_engine.get_alert_with_event(aid)
                    if alert_info:
                        status = alert_info.get("status", "ACTIVE")
                except Exception:
                    pass
                report.append(f"- **{aid}** (Status: `{status}`)")
        else:
            report.append("- No operational alerts matched.")
        report.append("")
        
        report.append("### Evidence References")
        if unique_evidence:
            for ep in sorted(list(unique_evidence)):
                url_path = ep.replace('\\', '/')
                report.append(f"- [{Path(ep).name}](file:///{url_path})")
        else:
            report.append("- No visual evidence links matched.")
        report.append("")
        
        report.append("### Confidence / Source Count")
        report.append(f"- **Total Sourced Items**: {total_sources} / 5 searched")
        report.append(f"- **Average Match Confidence (Similarity)**: {avg_sim:.3f}")
        
        return "\n".join(report)
