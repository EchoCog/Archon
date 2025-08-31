"""
Home Assistant Integration Tools for Archon Agents with OpenCog Enhancement
Provides reusable tools for integrating with Home Assistant instances using cognitive reasoning.
"""

import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import sys
import os

# Import OpenCog components for cognitive reasoning
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.opencog import opencog

class CognitiveHomeAssistantAPI:
    """
    Enhanced Home Assistant API client with OpenCog cognitive capabilities.
    Provides methods to interact with Home Assistant via REST API while storing
    knowledge in AtomSpace for intelligent reasoning and pattern recognition.
    """
    
    def __init__(self, ha_url: str, ha_token: str, enable_cognition: bool = True):
        self.ha_url = ha_url.rstrip('/')
        self.ha_token = ha_token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json"
        }
        
        # Initialize OpenCog components for cognitive enhancement
        self.enable_cognition = enable_cognition
        if enable_cognition:
            self.atomspace = opencog.atomspace.AtomSpace()
            self.cogserver = opencog.cogserver.CogServer(self.atomspace)
            self.utilities = opencog.utilities.Utilities(self.atomspace)
            self._initialize_home_automation_knowledge()
        else:
            self.atomspace = None
            self.cogserver = None
            self.utilities = None
    
    def _initialize_home_automation_knowledge(self):
        """Initialize foundational knowledge structure for home automation in AtomSpace."""
        if not self.enable_cognition:
            return
            
        # Create core concepts
        home_concept = self.atomspace.add_node("ConceptNode", "HomeAutomation")
        device_concept = self.atomspace.add_node("ConceptNode", "SmartDevice")
        user_concept = self.atomspace.add_node("ConceptNode", "User")
        pattern_concept = self.atomspace.add_node("ConceptNode", "UsagePattern")
        energy_concept = self.atomspace.add_node("ConceptNode", "EnergyEfficiency")
        
        # Establish foundational relationships
        self.utilities.create_atomese_expression(f"(Inheritance {device_concept} {home_concept})")
        self.utilities.create_atomese_expression(f"(Inheritance {pattern_concept} {home_concept})")
        self.utilities.create_atomese_expression(f"(Inheritance {energy_concept} {home_concept})")
        self.utilities.create_atomese_expression(f"(Evaluation (Predicate \"controls\") (List {user_concept} {device_concept}))")
        
        # Create domain-specific device categories
        for domain in ['light', 'climate', 'sensor', 'switch', 'lock', 'camera']:
            domain_concept = self.atomspace.add_node("ConceptNode", f"{domain.title()}Device")
            self.utilities.create_atomese_expression(f"(Inheritance {domain_concept} {device_concept})")
    
    def _store_device_interaction(self, entity_id: str, action: str, parameters: Dict = None):
        """Store device interaction in AtomSpace for learning."""
        if not self.enable_cognition:
            return
            
        timestamp = datetime.now().isoformat()
        device_node = self.atomspace.add_node("ConceptNode", entity_id)
        action_node = self.atomspace.add_node("ConceptNode", f"Action_{action}")
        time_node = self.atomspace.add_node("TimeNode", timestamp)
        
        # Store the interaction with temporal context
        self.utilities.create_atomese_expression(
            f"(AtTimeLink {time_node} (Evaluation (Predicate \"{action}\") (List {device_node} {action_node})))"
        )
        
        # Store parameters as attributes
        if parameters:
            for key, value in parameters.items():
                param_node = self.atomspace.add_node("ConceptNode", f"{key}_{value}")
                self.utilities.create_atomese_expression(
                    f"(Evaluation (Predicate \"has_parameter\") (List {action_node} {param_node}))"
                )
    
    def _get_cognitive_insights(self, entity_id: str, context: str) -> List[str]:
        """Generate cognitive insights about device or interaction."""
        if not self.enable_cognition:
            return []
        
        def insight_reasoner(atomspace, context_data, utilities):
            insights = []
            entity_domain = entity_id.split('.')[0] if '.' in entity_id else 'unknown'
            
            # Domain-specific insights
            if entity_domain == 'light':
                insights.append("💡 Light control detected - consider motion-based automation")
                insights.append("🌙 Tip: Smart scheduling can reduce energy consumption")
            elif entity_domain == 'climate':
                insights.append("🌡️ Climate control - optimize based on occupancy patterns")
                insights.append("📊 Energy savings opportunity through smart scheduling")
            elif entity_domain == 'sensor':
                insights.append("📈 Sensor data - valuable for automation triggers")
                insights.append("🤖 Consider incorporating into smart scenes")
            elif entity_domain == 'security' or 'lock' in entity_id or 'alarm' in entity_id:
                insights.append("🔒 Security device - important for safety automation")
                insights.append("📱 Consider mobile notifications for status changes")
            
            # Time-based insights
            current_hour = datetime.now().hour
            if 22 <= current_hour or current_hour <= 6:
                insights.append("🌙 Night time operation - consider sleep-friendly settings")
            elif 6 < current_hour <= 9:
                insights.append("🌅 Morning routine - good time for automated scenes")
            elif 17 <= current_hour < 22:
                insights.append("🌆 Evening time - consider comfort optimizations")
            
            return insights
        
        try:
            self.utilities.register_reasoner("cognitive_insights", insight_reasoner)
            return self.utilities.apply_reasoner("cognitive_insights", {"entity_id": entity_id, "context": context})
        except:
            return ["🧠 Cognitive analysis temporarily unavailable"]
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def call_service(self, domain: str, service: str, entity_id: Optional[str] = None, **service_data) -> Dict:
        """
        Call a Home Assistant service with cognitive enhancement.
        
        Args:
            domain: Service domain (e.g., 'light', 'climate')
            service: Service name (e.g., 'turn_on', 'set_temperature')
            entity_id: Target entity ID
            **service_data: Additional service parameters
        
        Returns:
            Dict containing the service call result with cognitive insights
        """
        await self._ensure_session()
        
        url = f"{self.ha_url}/api/services/{domain}/{service}"
        data = {}
        if entity_id:
            data["entity_id"] = entity_id
        data.update(service_data)
        
        # Store interaction in cognitive system
        self._store_device_interaction(entity_id or f"{domain}.{service}", service, service_data)
        
        async with self.session.post(url, headers=self.headers, json=data) as response:
            if response.status == 200:
                result_data = await response.json()
                
                # Generate cognitive insights
                insights = self._get_cognitive_insights(entity_id or f"{domain}.{service}", f"Service call: {domain}.{service}")
                
                return {
                    "success": True, 
                    "data": result_data,
                    "cognitive_insights": insights
                }
            else:
                error_text = await response.text()
                return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
    
    async def get_state(self, entity_id: str) -> Dict:
        """
        Get the state of a specific entity with cognitive analysis.
        
        Args:
            entity_id: The entity ID to query
        
        Returns:
            Dict containing entity state, attributes, and cognitive insights
        """
        await self._ensure_session()
        
        url = f"{self.ha_url}/api/states/{entity_id}"
        
        async with self.session.get(url, headers=self.headers) as response:
            if response.status == 200:
                entity_data = await response.json()
                
                # Store entity state in AtomSpace
                if self.enable_cognition:
                    entity_node = self.atomspace.add_node("ConceptNode", entity_id)
                    state_node = self.atomspace.add_node("ConceptNode", f"State_{entity_data.get('state', 'unknown')}")
                    timestamp_node = self.atomspace.add_node("TimeNode", entity_data.get('last_changed', datetime.now().isoformat()))
                    
                    self.utilities.create_atomese_expression(
                        f"(AtTimeLink {timestamp_node} (Evaluation (Predicate \"has_state\") (List {entity_node} {state_node})))"
                    )
                
                # Generate cognitive insights
                insights = self._get_cognitive_insights(entity_id, f"Entity state: {entity_data.get('state')}")
                
                return {
                    "success": True, 
                    "data": entity_data,
                    "cognitive_insights": insights
                }
            else:
                return {"success": False, "error": f"HTTP {response.status}"}
    
    async def get_states(self, domain: Optional[str] = None) -> Dict:
        """
        Get all entity states, optionally filtered by domain.
        
        Args:
            domain: Optional domain filter (e.g., 'light', 'sensor')
        
        Returns:
            Dict containing all matching entity states
        """
        await self._ensure_session()
        
        url = f"{self.ha_url}/api/states"
        
        async with self.session.get(url, headers=self.headers) as response:
            if response.status == 200:
                data = await response.json()
                if domain:
                    filtered_data = [
                        entity for entity in data 
                        if entity['entity_id'].startswith(f"{domain}.")
                    ]
                    return {"success": True, "data": filtered_data}
                return {"success": True, "data": data}
            else:
                return {"success": False, "error": f"HTTP {response.status}"}
    
    async def get_history(self, entity_id: str, hours: int = 24) -> Dict:
        """
        Get historical data for an entity.
        
        Args:
            entity_id: Entity to get history for
            hours: Number of hours of history to retrieve
        
        Returns:
            Dict containing historical state data
        """
        await self._ensure_session()
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        url = f"{self.ha_url}/api/history/period/{start_time.isoformat()}"
        params = {"filter_entity_id": entity_id}
        
        async with self.session.get(url, headers=self.headers, params=params) as response:
            if response.status == 200:
                return {"success": True, "data": await response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status}"}
    
    async def get_config(self) -> Dict:
        """
        Get Home Assistant configuration and system info.
        
        Returns:
            Dict containing system configuration
        """
        await self._ensure_session()
        
        url = f"{self.ha_url}/api/config"
        
        async with self.session.get(url, headers=self.headers) as response:
            if response.status == 200:
                return {"success": True, "data": await response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status}"}
    
    async def analyze_cognitive_patterns(self, entity_id: str, hours: int = 24) -> Dict:
        """
        Perform advanced cognitive pattern analysis using OpenCog reasoning.
        
        Args:
            entity_id: Entity to analyze patterns for
            hours: Number of hours of history to analyze
        
        Returns:
            Dict containing comprehensive cognitive analysis and predictions
        """
        if not self.enable_cognition:
            return {"success": False, "error": "Cognitive features not enabled"}
        
        # Get historical data
        history_result = await self.get_history(entity_id, hours)
        if not history_result["success"]:
            return history_result
        
        history_data = history_result["data"]
        if not history_data or not history_data[0]:
            return {"success": False, "error": "No historical data available"}
        
        entity_history = history_data[0]
        
        # Advanced cognitive pattern analysis
        def cognitive_pattern_reasoner(atomspace, analysis_data, utilities):
            analysis = {
                "temporal_patterns": {},
                "behavioral_insights": [],
                "predictions": [],
                "optimization_suggestions": [],
                "energy_efficiency": {},
                "automation_recommendations": []
            }
            
            entity_domain = entity_id.split('.')[0]
            states = [entry.get('state') for entry in entity_history]
            unique_states = set(states)
            
            # Temporal analysis
            hourly_activity = {}
            daily_activity = {}
            state_transitions = []
            
            for i, entry in enumerate(entity_history):
                timestamp = datetime.fromisoformat(entry.get('last_changed', entry.get('last_updated', '')).replace('Z', '+00:00'))
                state = entry.get('state', 'unknown')
                
                hour = timestamp.hour
                day = timestamp.strftime("%A")
                
                if hour not in hourly_activity:
                    hourly_activity[hour] = []
                hourly_activity[hour].append(state)
                
                if day not in daily_activity:
                    daily_activity[day] = []
                daily_activity[day].append(state)
                
                if i > 0:
                    prev_state = entity_history[i-1].get('state', 'unknown')
                    if prev_state != state:
                        state_transitions.append((prev_state, state, timestamp))
            
            # Peak activity analysis
            hour_activity_count = {hour: len([s for s in states_in_hour if s == 'on']) 
                                 for hour, states_in_hour in hourly_activity.items()}
            
            if hour_activity_count:
                peak_hour = max(hour_activity_count.items(), key=lambda x: x[1])
                analysis["temporal_patterns"]["peak_hour"] = peak_hour[0]
                analysis["temporal_patterns"]["peak_activity"] = peak_hour[1]
            
            # Behavioral insights
            current_hour = datetime.now().hour
            if entity_domain == 'light':
                on_states = len([s for s in states if s == 'on'])
                usage_ratio = on_states / len(states) if states else 0
                analysis["behavioral_insights"].append(f"Usage ratio: {usage_ratio:.2%}")
                
                if usage_ratio > 0.7:
                    analysis["optimization_suggestions"].append("High usage - consider smart dimming")
                elif usage_ratio < 0.1:
                    analysis["optimization_suggestions"].append("Low usage - consider motion automation")
                
                # Energy efficiency analysis
                analysis["energy_efficiency"]["usage_score"] = min(10, 10 - (usage_ratio * 5))
                
            elif entity_domain == 'climate':
                analysis["behavioral_insights"].append("Climate system monitoring comfort patterns")
                analysis["optimization_suggestions"].append("Schedule based on occupancy for energy savings")
                
            elif entity_domain == 'sensor':
                if 'motion' in entity_id.lower():
                    motion_events = len([s for s in states if s == 'on'])
                    analysis["behavioral_insights"].append(f"Motion events: {motion_events}")
                    if motion_events > 20:
                        analysis["automation_recommendations"].append("High traffic - optimize automation triggers")
            
            # Predictive analysis
            if states and current_hour in hour_activity_count:
                current_activity = hour_activity_count[current_hour]
                if current_activity > 0:
                    analysis["predictions"].append("Activity expected in current time window")
                else:
                    analysis["predictions"].append("Low activity period - energy saving opportunity")
            
            # Automation recommendations
            if len(state_transitions) > 5:
                analysis["automation_recommendations"].append("Frequent state changes - consider smart automation")
            
            # Context-aware suggestions
            if 22 <= current_hour or current_hour <= 6:
                analysis["optimization_suggestions"].append("Night mode - minimize disruptions")
            elif 6 < current_hour <= 9:
                analysis["automation_recommendations"].append("Morning routine automation opportunity")
            
            return analysis
        
        try:
            self.utilities.register_reasoner("cognitive_pattern_analysis", cognitive_pattern_reasoner)
            cognitive_analysis = self.utilities.apply_reasoner("cognitive_pattern_analysis", {"history": entity_history})
            
            return {
                "success": True,
                "entity_id": entity_id,
                "analysis_period": f"{hours} hours",
                "total_state_changes": len(entity_history),
                "cognitive_analysis": cognitive_analysis,
                "knowledge_atoms": len(self.atomspace.atoms) if self.atomspace else 0
            }
        except Exception as e:
            return {"success": False, "error": f"Cognitive analysis failed: {str(e)}"}

class CognitiveHomeAssistantTools:
    """
    Enhanced tools for Home Assistant operations with OpenCog cognitive capabilities.
    These tools provide intelligent automation suggestions and pattern recognition.
    """
    
    def __init__(self, ha_api: CognitiveHomeAssistantAPI):
        self.ha_api = ha_api
    
    async def control_lights(self, action: str, entity_ids: List[str] = None, **kwargs) -> str:
        """
        Control lighting devices with natural language actions.
        
        Args:
            action: Action to perform ('on', 'off', 'dim', 'brighten')
            entity_ids: List of light entity IDs (if None, affects all lights)
            **kwargs: Additional parameters (brightness, color, etc.)
        """
        if entity_ids is None:
            # Get all lights
            states_result = await self.ha_api.get_states(domain="light")
            if not states_result["success"]:
                return f"❌ Error getting lights: {states_result['error']}"
            
            entity_ids = [entity["entity_id"] for entity in states_result["data"]]
        
        if action.lower() in ['on', 'turn_on', 'turn on']:
            service = 'turn_on'
        elif action.lower() in ['off', 'turn_off', 'turn off']:
            service = 'turn_off'
        else:
            return f"❌ Unknown action: {action}"
        
        results = []
        for entity_id in entity_ids:
            result = await self.ha_api.call_service('light', service, entity_id, **kwargs)
            if result["success"]:
                results.append(f"✅ {entity_id}: {action}")
            else:
                results.append(f"❌ {entity_id}: {result['error']}")
        
        return "\n".join(results)
    
    async def control_climate(self, temperature: float = None, mode: str = None, entity_ids: List[str] = None) -> str:
        """
        Control climate/HVAC devices.
        
        Args:
            temperature: Target temperature
            mode: HVAC mode ('heat', 'cool', 'auto', 'off')
            entity_ids: List of climate entity IDs
        """
        if entity_ids is None:
            # Get all climate entities
            states_result = await self.ha_api.get_states(domain="climate")
            if not states_result["success"]:
                return f"❌ Error getting climate devices: {states_result['error']}"
            
            entity_ids = [entity["entity_id"] for entity in states_result["data"]]
        
        results = []
        for entity_id in entity_ids:
            if temperature is not None:
                result = await self.ha_api.call_service('climate', 'set_temperature', entity_id, temperature=temperature)
                if result["success"]:
                    results.append(f"🌡️ {entity_id}: temperature set to {temperature}°")
                else:
                    results.append(f"❌ {entity_id}: {result['error']}")
            
            if mode is not None:
                result = await self.ha_api.call_service('climate', 'set_hvac_mode', entity_id, hvac_mode=mode)
                if result["success"]:
                    results.append(f"🏠 {entity_id}: mode set to {mode}")
                else:
                    results.append(f"❌ {entity_id}: {result['error']}")
        
        return "\n".join(results)
    
    async def get_security_status(self) -> str:
        """
        Get status of security-related devices (locks, alarms, doors, windows).
        """
        security_domains = ['lock', 'alarm_control_panel', 'binary_sensor']
        security_entities = []
        
        for domain in security_domains:
            states_result = await self.ha_api.get_states(domain=domain)
            if states_result["success"]:
                for entity in states_result["data"]:
                    if any(keyword in entity["entity_id"] for keyword in 
                          ['door', 'window', 'lock', 'alarm', 'motion', 'security']):
                        security_entities.append(entity)
        
        if not security_entities:
            return "🔒 No security devices found"
        
        result = "🔒 Security Status:\n"
        for entity in security_entities:
            state = entity.get("state", "unknown")
            result += f"• {entity['entity_id']}: {state}\n"
        
        return result
    
    async def analyze_usage_patterns(self, entity_id: str, days: int = 7) -> str:
        """
        Analyze usage patterns for a specific entity over multiple days.
        
        Args:
            entity_id: Entity to analyze
            days: Number of days to analyze
        """
        history_result = await self.ha_api.get_history(entity_id, hours=days * 24)
        
        if not history_result["success"]:
            return f"❌ Error getting history: {history_result['error']}"
        
        history_data = history_result["data"]
        if not history_data or not history_data[0]:
            return f"📊 No usage data found for {entity_id}"
        
        entity_history = history_data[0]
        
        # Analyze patterns
        state_changes = []
        for entry in entity_history:
            timestamp = datetime.fromisoformat(entry["last_changed"].replace("Z", "+00:00"))
            state = entry["state"]
            state_changes.append((timestamp, state))
        
        # Basic pattern analysis
        daily_usage = {}
        hourly_usage = {}
        
        for timestamp, state in state_changes:
            day = timestamp.strftime("%A")
            hour = timestamp.hour
            
            if day not in daily_usage:
                daily_usage[day] = 0
            if hour not in hourly_usage:
                hourly_usage[hour] = 0
            
            if state == "on":
                daily_usage[day] += 1
                hourly_usage[hour] += 1
        
        result = f"📊 Usage Pattern Analysis for {entity_id} (last {days} days):\n\n"
        
        # Daily patterns
        result += "📅 Daily Usage:\n"
        for day, count in sorted(daily_usage.items(), key=lambda x: x[1], reverse=True):
            result += f"• {day}: {count} activations\n"
        
        # Peak hours
        result += "\n⏰ Peak Hours:\n"
        sorted_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)
        for hour, count in sorted_hours[:5]:
            result += f"• {hour:02d}:00: {count} activations\n"
        
        # Recommendations
        result += "\n💡 Optimization Suggestions:\n"
        peak_hour = sorted_hours[0][0] if sorted_hours else None
        if peak_hour is not None:
            result += f"• Peak usage at {peak_hour:02d}:00 - consider automation triggers\n"
        
        most_active_day = max(daily_usage.items(), key=lambda x: x[1])[0] if daily_usage else None
        if most_active_day:
            result += f"• Most active on {most_active_day} - optimize for this day\n"
        
        return result

# Utility functions for agent integration
def create_ha_deps_class():
    """
    Create a Pydantic model for Home Assistant dependencies.
    Use this in your agent's deps_type parameter.
    """
    class HADeps(BaseModel):
        ha_url: str
        ha_token: str
        ha_api: Optional[HomeAssistantAPI] = None
        ha_tools: Optional[HomeAssistantTools] = None
        
        def model_post_init(self, __context):
            if not self.ha_api:
                self.ha_api = HomeAssistantAPI(self.ha_url, self.ha_token)
            if not self.ha_tools:
                self.ha_tools = HomeAssistantTools(self.ha_api)
    
    return HADeps

# Example configuration for common Home Assistant setups
HA_EXAMPLE_CONFIG = {
    "typical_entities": {
        "lights": [
            "light.living_room",
            "light.kitchen", 
            "light.bedroom",
            "light.bathroom"
        ],
        "climate": [
            "climate.main_floor",
            "climate.upstairs"
        ],
        "sensors": [
            "sensor.temperature",
            "sensor.humidity",
            "sensor.energy_usage"
        ],
        "security": [
            "lock.front_door",
            "binary_sensor.motion_living_room",
            "alarm_control_panel.home"
        ]
    },
    "common_services": {
        "light.turn_on": {"brightness": 255, "color_name": "white"},
        "light.turn_off": {},
        "climate.set_temperature": {"temperature": 72},
        "climate.set_hvac_mode": {"hvac_mode": "auto"},
        "lock.lock": {},
        "lock.unlock": {},
        "automation.trigger": {},
        "script.turn_on": {}
    }
}