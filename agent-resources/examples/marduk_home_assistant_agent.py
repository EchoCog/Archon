"""
Marduk's Lab - Home Assistant Integration Agent
A comprehensive AI agent for intelligent home automation control

This agent demonstrates the integration between Archon and Home Assistant,
providing natural language control, pattern learning, and predictive automation.
"""

import asyncio
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

# Dependencies for the Home Assistant agent
class HADeps(BaseModel):
    ha_url: str
    ha_token: str
    session: Optional[aiohttp.ClientSession] = None

# Home Assistant Control Agent
marduk_agent = Agent(
    'claude-3-5-sonnet-20241022',
    deps_type=HADeps,
    system_prompt="""
    You are Marduk, the intelligent home automation assistant powered by Archon.
    You can control all smart home devices through Home Assistant, learn from usage patterns, 
    and proactively suggest improvements to make the home more comfortable and efficient.
    
    Core capabilities:
    - Natural language device control ("turn off all lights", "set temperature to 72")
    - Real-time device status monitoring and reporting
    - Pattern recognition and usage analysis
    - Predictive automation suggestions
    - Energy optimization recommendations
    - Security monitoring and alerts
    - Scene and automation management
    
    You have access to the complete Home Assistant API and can:
    1. Call any service (light.turn_on, climate.set_temperature, etc.)
    2. Query device states and attributes
    3. Access historical data for pattern analysis
    4. Create and modify automations
    5. Manage scenes and scripts
    6. Monitor system health and performance
    
    Always respond with helpful information about what actions were taken and current system status.
    If you detect patterns or optimization opportunities, proactively suggest improvements.
    
    For device control, always confirm the action and provide current status.
    For queries, provide comprehensive information including related devices and suggestions.
    """
)

@marduk_agent.tool
async def call_ha_service(
    ctx: RunContext[HADeps], 
    domain: str, 
    service: str, 
    entity_id: Optional[str] = None,
    **service_data
) -> str:
    """
    Call a Home Assistant service to control devices or execute actions.
    
    Args:
        domain: The domain of the service (e.g., 'light', 'climate', 'switch')
        service: The service to call (e.g., 'turn_on', 'turn_off', 'set_temperature')
        entity_id: The entity ID to target (optional for some services)
        **service_data: Additional service data (e.g., brightness, color, temperature)
    
    Examples:
        - call_ha_service('light', 'turn_on', entity_id='light.living_room', brightness=255)
        - call_ha_service('climate', 'set_temperature', entity_id='climate.main', temperature=72)
        - call_ha_service('automation', 'trigger', entity_id='automation.good_night')
    """
    url = f"{ctx.deps.ha_url}/api/services/{domain}/{service}"
    headers = {
        "Authorization": f"Bearer {ctx.deps.ha_token}",
        "Content-Type": "application/json"
    }
    
    data = {}
    if entity_id:
        data["entity_id"] = entity_id
    data.update(service_data)
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return f"✅ Service {domain}.{service} called successfully. Response: {json.dumps(result, indent=2)}"
            else:
                error_text = await response.text()
                return f"❌ Error calling service {domain}.{service}: {response.status} - {error_text}"
    except Exception as e:
        return f"❌ Exception calling service {domain}.{service}: {str(e)}"

@marduk_agent.tool
async def get_entity_state(ctx: RunContext[HADeps], entity_id: str) -> str:
    """
    Get the current state and attributes of a Home Assistant entity.
    
    Args:
        entity_id: The entity ID to query (e.g., 'light.living_room', 'sensor.temperature')
    
    Returns detailed information about the entity including state, attributes, and last changed time.
    """
    url = f"{ctx.deps.ha_url}/api/states/{entity_id}"
    headers = {"Authorization": f"Bearer {ctx.deps.ha_token}"}
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return f"""
📊 Entity State: {entity_id}
State: {data.get('state', 'unknown')}
Last Changed: {data.get('last_changed', 'unknown')}
Last Updated: {data.get('last_updated', 'unknown')}

Attributes:
{json.dumps(data.get('attributes', {}), indent=2)}
"""
            else:
                return f"❌ Error getting state for {entity_id}: {response.status}"
    except Exception as e:
        return f"❌ Exception getting state for {entity_id}: {str(e)}"

@marduk_agent.tool
async def get_all_entities(ctx: RunContext[HADeps], domain: Optional[str] = None) -> str:
    """
    Get all entities or entities from a specific domain.
    
    Args:
        domain: Optional domain filter (e.g., 'light', 'sensor', 'switch')
    
    Returns a list of all entities with their current states.
    """
    url = f"{ctx.deps.ha_url}/api/states"
    headers = {"Authorization": f"Bearer {ctx.deps.ha_token}"}
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                
                if domain:
                    filtered_entities = [
                        entity for entity in data 
                        if entity['entity_id'].startswith(f"{domain}.")
                    ]
                else:
                    filtered_entities = data
                
                result = f"📋 Found {len(filtered_entities)} entities"
                if domain:
                    result += f" in domain '{domain}'"
                result += ":\n\n"
                
                for entity in filtered_entities[:20]:  # Limit to first 20 to avoid huge responses
                    result += f"• {entity['entity_id']}: {entity['state']}\n"
                
                if len(filtered_entities) > 20:
                    result += f"\n... and {len(filtered_entities) - 20} more entities"
                
                return result
            else:
                return f"❌ Error getting entities: {response.status}"
    except Exception as e:
        return f"❌ Exception getting entities: {str(e)}"

@marduk_agent.tool
async def get_entity_history(
    ctx: RunContext[HADeps], 
    entity_id: str, 
    hours: int = 24
) -> str:
    """
    Get historical data for an entity to analyze patterns and usage.
    
    Args:
        entity_id: The entity ID to get history for
        hours: Number of hours of history to retrieve (default: 24)
    
    Returns historical state changes for pattern analysis.
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    url = f"{ctx.deps.ha_url}/api/history/period/{start_time.isoformat()}"
    headers = {"Authorization": f"Bearer {ctx.deps.ha_token}"}
    params = {"filter_entity_id": entity_id}
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if not data or not data[0]:
                    return f"📈 No history found for {entity_id} in the last {hours} hours"
                
                history = data[0]  # First entity's history
                
                result = f"📈 History for {entity_id} (last {hours} hours):\n\n"
                
                for entry in history[-10:]:  # Show last 10 changes
                    timestamp = entry.get('last_changed', entry.get('last_updated', 'unknown'))
                    state = entry.get('state', 'unknown')
                    result += f"• {timestamp}: {state}\n"
                
                # Basic pattern analysis
                states = [entry.get('state') for entry in history if entry.get('state')]
                unique_states = set(states)
                
                result += f"\n📊 Pattern Analysis:\n"
                result += f"• Total state changes: {len(history)}\n"
                result += f"• Unique states: {', '.join(unique_states)}\n"
                result += f"• Most recent state: {states[-1] if states else 'unknown'}\n"
                
                return result
            else:
                return f"❌ Error getting history for {entity_id}: {response.status}"
    except Exception as e:
        return f"❌ Exception getting history for {entity_id}: {str(e)}"

@marduk_agent.tool
async def create_automation(
    ctx: RunContext[HADeps],
    automation_config: Dict[str, Any]
) -> str:
    """
    Create a new Home Assistant automation.
    
    Args:
        automation_config: The automation configuration as a dictionary
    
    Example automation_config:
    {
        "alias": "Turn on lights at sunset",
        "trigger": {
            "platform": "sun",
            "event": "sunset"
        },
        "action": {
            "service": "light.turn_on",
            "entity_id": "light.living_room"
        }
    }
    """
    url = f"{ctx.deps.ha_url}/api/config/automation/config"
    headers = {
        "Authorization": f"Bearer {ctx.deps.ha_token}",
        "Content-Type": "application/json"
    }
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.post(url, headers=headers, json=automation_config) as response:
            if response.status in [200, 201]:
                result = await response.json()
                return f"✅ Automation '{automation_config.get('alias', 'unnamed')}' created successfully. ID: {result.get('id')}"
            else:
                error_text = await response.text()
                return f"❌ Error creating automation: {response.status} - {error_text}"
    except Exception as e:
        return f"❌ Exception creating automation: {str(e)}"

@marduk_agent.tool
async def get_system_health(ctx: RunContext[HADeps]) -> str:
    """
    Get Home Assistant system health and performance information.
    
    Returns system status, component health, and performance metrics.
    """
    url = f"{ctx.deps.ha_url}/api/config"
    headers = {"Authorization": f"Bearer {ctx.deps.ha_token}"}
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.get(url, headers=headers) as response:
            if response.status == 200:
                config = await response.json()
                
                result = f"""
🏠 Home Assistant System Health
===============================
Version: {config.get('version', 'unknown')}
Location: {config.get('location_name', 'unknown')}
Time Zone: {config.get('time_zone', 'unknown')}
Elevation: {config.get('elevation', 'unknown')}m
Unit System: {config.get('unit_system', {}).get('length', 'unknown')}

Configuration:
• Internal URL: {config.get('internal_url', 'not set')}
• External URL: {config.get('external_url', 'not set')}
• Safe Mode: {config.get('safe_mode', False)}
• State: {config.get('state', 'unknown')}

Components: {len(config.get('components', []))} loaded
Whitelist External Dirs: {len(config.get('whitelist_external_dirs', []))} configured
"""
                return result
            else:
                return f"❌ Error getting system health: {response.status}"
    except Exception as e:
        return f"❌ Exception getting system health: {str(e)}"

@marduk_agent.tool
async def analyze_energy_usage(ctx: RunContext[HADeps]) -> str:
    """
    Analyze energy usage patterns and provide optimization suggestions.
    
    Looks at energy sensors and provides insights for energy savings.
    """
    # Get all energy-related sensors
    url = f"{ctx.deps.ha_url}/api/states"
    headers = {"Authorization": f"Bearer {ctx.deps.ha_token}"}
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                
                # Filter energy-related entities
                energy_entities = [
                    entity for entity in data 
                    if any(keyword in entity['entity_id'].lower() for keyword in 
                          ['energy', 'power', 'consumption', 'usage', 'watt', 'kwh'])
                ]
                
                result = f"⚡ Energy Usage Analysis\n"
                result += f"========================\n\n"
                result += f"Found {len(energy_entities)} energy-related entities:\n\n"
                
                for entity in energy_entities[:10]:  # Limit to first 10
                    state = entity.get('state', 'unknown')
                    unit = entity.get('attributes', {}).get('unit_of_measurement', '')
                    result += f"• {entity['entity_id']}: {state} {unit}\n"
                
                # Provide energy optimization suggestions
                result += f"\n💡 Optimization Suggestions:\n"
                result += f"• Monitor high power consumption devices\n"
                result += f"• Set up automations to turn off devices when not needed\n"
                result += f"• Use occupancy sensors for lighting automation\n"
                result += f"• Implement time-based heating/cooling schedules\n"
                result += f"• Consider smart plugs for vampire power elimination\n"
                
                return result
            else:
                return f"❌ Error analyzing energy usage: {response.status}"
    except Exception as e:
        return f"❌ Exception analyzing energy usage: {str(e)}"

# Example usage and testing
async def main():
    """
    Example usage of the Marduk Home Assistant agent.
    This would be called by Archon when the agent is deployed.
    """
    # Initialize dependencies
    deps = HADeps(
        ha_url="http://homeassistant.local:8123",  # Replace with your HA URL
        ha_token="your_long_lived_access_token"    # Replace with your token
    )
    
    # Example conversations
    examples = [
        "What's the current temperature in the living room?",
        "Turn on all the lights in the house",
        "Set the thermostat to 72 degrees",
        "Show me the energy usage for today",
        "Create an automation to turn off all lights at midnight",
        "What devices are currently on?",
        "Analyze the usage patterns for the main lights"
    ]
    
    print("🏠 Marduk's Lab - Home Assistant Agent Ready!")
    print("Example commands you can try:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

if __name__ == "__main__":
    asyncio.run(main())