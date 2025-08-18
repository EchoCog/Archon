"""
Home Assistant Integration Tools for Archon Agents
Provides reusable tools for integrating with Home Assistant instances.
"""

import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class HomeAssistantAPI:
    """
    Home Assistant API client for Archon agents.
    Provides methods to interact with Home Assistant via REST API.
    """
    
    def __init__(self, ha_url: str, ha_token: str):
        self.ha_url = ha_url.rstrip('/')
        self.ha_token = ha_token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json"
        }
    
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
        Call a Home Assistant service.
        
        Args:
            domain: Service domain (e.g., 'light', 'climate')
            service: Service name (e.g., 'turn_on', 'set_temperature')
            entity_id: Target entity ID
            **service_data: Additional service parameters
        
        Returns:
            Dict containing the service call result
        """
        await self._ensure_session()
        
        url = f"{self.ha_url}/api/services/{domain}/{service}"
        data = {}
        if entity_id:
            data["entity_id"] = entity_id
        data.update(service_data)
        
        async with self.session.post(url, headers=self.headers, json=data) as response:
            if response.status == 200:
                return {"success": True, "data": await response.json()}
            else:
                error_text = await response.text()
                return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
    
    async def get_state(self, entity_id: str) -> Dict:
        """
        Get the state of a specific entity.
        
        Args:
            entity_id: The entity ID to query
        
        Returns:
            Dict containing entity state and attributes
        """
        await self._ensure_session()
        
        url = f"{self.ha_url}/api/states/{entity_id}"
        
        async with self.session.get(url, headers=self.headers) as response:
            if response.status == 200:
                return {"success": True, "data": await response.json()}
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

class HomeAssistantTools:
    """
    High-level tools for common Home Assistant operations.
    These can be used as decorators in Pydantic AI agents.
    """
    
    def __init__(self, ha_api: HomeAssistantAPI):
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