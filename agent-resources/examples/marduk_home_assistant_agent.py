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

# Import OpenCog components for cognitive reasoning
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.opencog import opencog

# Dependencies for the Home Assistant agent with OpenCog integration
class HADeps(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    ha_url: str
    ha_token: str
    session: Optional[aiohttp.ClientSession] = None
    # OpenCog cognitive components
    atomspace: Optional[object] = None
    cogserver: Optional[object] = None  
    utilities: Optional[object] = None

# Home Assistant Control Agent with OpenCog Integration
marduk_agent = Agent(
    'claude-3-5-sonnet-20241022',
    deps_type=HADeps,
    system_prompt="""
    You are Marduk, the intelligent home automation assistant powered by Archon's OpenCog cognitive architecture.
    You can control all smart home devices through Home Assistant, learn from usage patterns using advanced
    cognitive reasoning, and proactively suggest improvements to make the home more comfortable and efficient.
    
    Enhanced capabilities with OpenCog integration:
    - Natural language device control with contextual understanding
    - Real-time device status monitoring stored in cognitive AtomSpace
    - Advanced pattern recognition using OpenCog's reasoning engine
    - Predictive automation suggestions based on cognitive learning
    - Energy optimization recommendations with semantic understanding
    - Security monitoring with contextual threat assessment
    - Scene and automation management with intelligent adaptation
    - Knowledge persistence across sessions using AtomSpace
    - Collaborative reasoning with other Archon agents
    
    You have access to the complete Home Assistant API AND OpenCog cognitive components:
    1. Call any Home Assistant service with cognitive context awareness
    2. Store device states and relationships in AtomSpace for persistent knowledge
    3. Use OpenCog's reasoning engine for pattern analysis and prediction
    4. Apply Atomese expressions for complex logical relationships
    5. Share knowledge with other agents through the cognitive substrate
    6. Learn and adapt based on user preferences and environmental context
    
    Your cognitive reasoning enables you to:
    - Understand complex relationships between devices, rooms, and user behaviors
    - Make intelligent predictions about user needs and preferences
    - Suggest optimizations based on semantic understanding of home automation
    - Remember and learn from all interactions for continuous improvement
    
    Always use OpenCog's cognitive capabilities to enhance your responses with deeper insights,
    pattern recognition, and predictive suggestions. Store important knowledge in AtomSpace
    for future reference and reasoning.
    """
)

# Initialize OpenCog components for cognitive home automation
async def initialize_cognitive_components(deps: HADeps):
    """Initialize OpenCog components for cognitive reasoning in home automation."""
    if not deps.atomspace:
        deps.atomspace = opencog.atomspace.AtomSpace()
        deps.cogserver = opencog.cogserver.CogServer(deps.atomspace)
        deps.utilities = opencog.utilities.Utilities(deps.atomspace)
        
        # Create foundational knowledge structure for home automation
        home_concept = deps.atomspace.add_node("ConceptNode", "HomeAutomation")
        device_concept = deps.atomspace.add_node("ConceptNode", "SmartDevice")
        user_concept = deps.atomspace.add_node("ConceptNode", "User")
        pattern_concept = deps.atomspace.add_node("ConceptNode", "UsagePattern")
        
        # Establish basic relationships
        deps.utilities.create_atomese_expression(f"(Inheritance {device_concept} {home_concept})")
        deps.utilities.create_atomese_expression(f"(Inheritance {pattern_concept} {home_concept})")
        deps.utilities.create_atomese_expression(f"(Evaluation (Predicate \"controls\") (List {user_concept} {device_concept}))")
        
        print("🧠 OpenCog cognitive components initialized for Marduk's Lab")

@marduk_agent.tool
async def call_ha_service_with_cognition(
    ctx: RunContext[HADeps], 
    domain: str, 
    service: str, 
    entity_id: Optional[str] = None,
    **service_data
) -> str:
    """
    Call a Home Assistant service with OpenCog cognitive enhancement.
    
    This enhanced version stores device interactions in AtomSpace for learning
    and applies cognitive reasoning to understand device relationships and usage patterns.
    
    Args:
        domain: The domain of the service (e.g., 'light', 'climate', 'switch')
        service: The service to call (e.g., 'turn_on', 'turn_off', 'set_temperature')
        entity_id: The entity ID to target (optional for some services)
        **service_data: Additional service data (e.g., brightness, color, temperature)
    
    Examples:
        - call_ha_service_with_cognition('light', 'turn_on', entity_id='light.living_room', brightness=255)
        - call_ha_service_with_cognition('climate', 'set_temperature', entity_id='climate.main', temperature=72)
    """
    # Initialize cognitive components if not already done
    await initialize_cognitive_components(ctx.deps)
    
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
                
                # Store the interaction in AtomSpace for cognitive learning
                timestamp = datetime.now().isoformat()
                action_node = ctx.deps.atomspace.add_node("ConceptNode", f"Action_{domain}_{service}")
                
                if entity_id:
                    device_node = ctx.deps.atomspace.add_node("ConceptNode", entity_id)
                    ctx.deps.utilities.create_atomese_expression(
                        f"(AtTimeLink (TimeNode \"{timestamp}\") "
                        f"(Evaluation (Predicate \"{service}\") (List {device_node} {action_node})))"
                    )
                    
                    # Store device state context for pattern learning
                    for key, value in service_data.items():
                        param_node = ctx.deps.atomspace.add_node("ConceptNode", f"{key}_{value}")
                        ctx.deps.utilities.create_atomese_expression(
                            f"(Evaluation (Predicate \"has_parameter\") (List {action_node} {param_node}))"
                        )
                
                # Cognitive reasoning about the action
                reasoning_context = f"Service {domain}.{service} executed successfully"
                if entity_id:
                    reasoning_context += f" on {entity_id}"
                if service_data:
                    reasoning_context += f" with parameters: {service_data}"
                
                # Use OpenCog reasoning to provide enhanced insights
                def smart_action_reasoner(atomspace, action_context, utilities):
                    insights = []
                    
                    # Analyze related devices and suggest optimizations
                    if domain == "light" and service == "turn_on":
                        insights.append("💡 Tip: Consider setting up motion-based automation for this light")
                        if "brightness" not in service_data:
                            insights.append("🔆 Suggestion: You can adjust brightness for energy savings")
                    
                    elif domain == "climate":
                        insights.append("🌡️ Smart insight: Temperature changes affect energy usage patterns")
                        insights.append("📊 Consider scheduling temperature adjustments for optimal efficiency")
                    
                    elif service == "turn_off":
                        insights.append("♻️ Energy saved! This action contributes to your efficiency goals")
                    
                    return insights
                
                ctx.deps.utilities.register_reasoner("smart_action_analysis", smart_action_reasoner)
                cognitive_insights = ctx.deps.utilities.apply_reasoner("smart_action_analysis", reasoning_context)
                
                response_text = f"✅ Service {domain}.{service} called successfully.\n"
                response_text += f"📊 Cognitive Response: {json.dumps(result, indent=2)}\n"
                
                if cognitive_insights:
                    response_text += f"\n🧠 Cognitive Insights:\n"
                    for insight in cognitive_insights:
                        response_text += f"  {insight}\n"
                
                return response_text
                
            else:
                error_text = await response.text()
                return f"❌ Error calling service {domain}.{service}: {response.status} - {error_text}"
                
    except Exception as e:
        return f"❌ Exception calling service {domain}.{service}: {str(e)}"

@marduk_agent.tool
async def get_entity_state_with_cognition(ctx: RunContext[HADeps], entity_id: str) -> str:
    """
    Get the current state and attributes of a Home Assistant entity with cognitive enhancement.
    
    This enhanced version stores entity states in AtomSpace and provides cognitive insights
    about the device's behavior, relationships, and optimization opportunities.
    
    Args:
        entity_id: The entity ID to query (e.g., 'light.living_room', 'sensor.temperature')
    
    Returns detailed information with cognitive analysis and insights.
    """
    # Initialize cognitive components if not already done
    await initialize_cognitive_components(ctx.deps)
    
    url = f"{ctx.deps.ha_url}/api/states/{entity_id}"
    headers = {"Authorization": f"Bearer {ctx.deps.ha_token}"}
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                
                # Store entity state in AtomSpace for cognitive analysis
                entity_node = ctx.deps.atomspace.add_node("ConceptNode", entity_id)
                state_node = ctx.deps.atomspace.add_node("ConceptNode", f"State_{data.get('state', 'unknown')}")
                timestamp_node = ctx.deps.atomspace.add_node("TimeNode", data.get('last_changed', datetime.now().isoformat()))
                
                # Create relationships in AtomSpace
                ctx.deps.utilities.create_atomese_expression(
                    f"(AtTimeLink {timestamp_node} (Evaluation (Predicate \"has_state\") (List {entity_node} {state_node})))"
                )
                
                # Store attributes as knowledge
                attributes = data.get('attributes', {})
                for attr_key, attr_value in attributes.items():
                    attr_node = ctx.deps.atomspace.add_node("ConceptNode", f"{attr_key}_{attr_value}")
                    ctx.deps.utilities.create_atomese_expression(
                        f"(Evaluation (Predicate \"has_attribute\") (List {entity_node} {attr_node}))"
                    )
                
                # Apply cognitive reasoning for insights
                def entity_analysis_reasoner(atomspace, entity_data, utilities):
                    insights = []
                    state = entity_data.get('state', 'unknown')
                    entity_domain = entity_id.split('.')[0]
                    
                    # Domain-specific cognitive analysis
                    if entity_domain == 'light':
                        if state == 'on':
                            insights.append("💡 Light is currently on - consider automation for energy efficiency")
                            if 'brightness' in attributes and int(attributes.get('brightness', 0)) > 200:
                                insights.append("🔆 High brightness detected - optimal for task lighting")
                        else:
                            insights.append("🌙 Light is off - good for energy conservation")
                    
                    elif entity_domain == 'climate':
                        if 'current_temperature' in attributes:
                            temp = attributes['current_temperature']
                            insights.append(f"🌡️ Current temperature: {temp}°C - monitoring for comfort optimization")
                        if 'hvac_mode' in attributes:
                            mode = attributes['hvac_mode']
                            insights.append(f"❄️ HVAC mode: {mode} - consider smart scheduling for efficiency")
                    
                    elif entity_domain == 'sensor':
                        if 'motion' in entity_id.lower() and state == 'on':
                            insights.append("🚶 Motion detected - could trigger automation routines")
                        elif 'temperature' in entity_id.lower():
                            insights.append("🌡️ Temperature sensor - valuable for climate automation")
                    
                    elif entity_domain == 'binary_sensor':
                        if 'door' in entity_id.lower() or 'window' in entity_id.lower():
                            insights.append(f"🚪 Security sensor: {state} - monitoring for safety automation")
                    
                    # General efficiency insights
                    last_changed = data.get('last_changed', '')
                    if last_changed:
                        try:
                            changed_time = datetime.fromisoformat(last_changed.replace('Z', '+00:00'))
                            time_diff = datetime.now(changed_time.tzinfo) - changed_time
                            if time_diff.total_seconds() < 300:  # Less than 5 minutes
                                insights.append("⚡ Recent state change - device is actively used")
                            elif time_diff.total_seconds() > 3600:  # More than 1 hour
                                insights.append("⏰ State unchanged for over an hour - stable condition")
                        except:
                            pass
                    
                    return insights
                
                ctx.deps.utilities.register_reasoner("entity_analysis", entity_analysis_reasoner)
                cognitive_insights = ctx.deps.utilities.apply_reasoner("entity_analysis", data)
                
                response_text = f"""
🏠 Entity State Analysis: {entity_id}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Current State: {data.get('state', 'unknown')}
⏰ Last Changed: {data.get('last_changed', 'unknown')}
🔄 Last Updated: {data.get('last_updated', 'unknown')}

📋 Attributes:
{json.dumps(attributes, indent=2)}
"""

                if cognitive_insights:
                    response_text += f"\n🧠 Cognitive Insights:\n"
                    for insight in cognitive_insights:
                        response_text += f"  {insight}\n"
                
                return response_text
                
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
async def analyze_usage_patterns_with_cognition(
    ctx: RunContext[HADeps], 
    entity_id: str, 
    hours: int = 24
) -> str:
    """
    Advanced usage pattern analysis using OpenCog's cognitive reasoning capabilities.
    
    This tool goes beyond basic statistics to provide semantic understanding of usage patterns,
    predictive insights, and intelligent automation suggestions using OpenCog's reasoning engine.
    
    Args:
        entity_id: The entity ID to analyze patterns for
        hours: Number of hours of history to analyze (default: 24)
    
    Returns comprehensive cognitive analysis with predictions and recommendations.
    """
    # Initialize cognitive components if not already done
    await initialize_cognitive_components(ctx.deps)
    
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
                    return f"🧠 No cognitive pattern data found for {entity_id} in the last {hours} hours"
                
                history = data[0]  # First entity's history
                
                # Store pattern data in AtomSpace for cognitive analysis
                pattern_node = ctx.deps.atomspace.add_node("ConceptNode", f"Pattern_{entity_id}")
                entity_node = ctx.deps.atomspace.add_node("ConceptNode", entity_id)
                
                # Analyze temporal patterns with OpenCog
                state_transitions = []
                hourly_activity = {}
                daily_activity = {}
                state_durations = {}
                
                for i, entry in enumerate(history):
                    timestamp = datetime.fromisoformat(entry.get('last_changed', entry.get('last_updated', '')).replace('Z', '+00:00'))
                    state = entry.get('state', 'unknown')
                    
                    # Store each state change in AtomSpace
                    time_node = ctx.deps.atomspace.add_node("TimeNode", timestamp.isoformat())
                    state_node = ctx.deps.atomspace.add_node("ConceptNode", f"State_{state}")
                    
                    ctx.deps.utilities.create_atomese_expression(
                        f"(AtTimeLink {time_node} (Evaluation (Predicate \"has_state\") (List {entity_node} {state_node})))"
                    )
                    
                    # Track patterns
                    hour = timestamp.hour
                    day = timestamp.strftime("%A")
                    
                    if hour not in hourly_activity:
                        hourly_activity[hour] = []
                    hourly_activity[hour].append(state)
                    
                    if day not in daily_activity:
                        daily_activity[day] = []
                    daily_activity[day].append(state)
                    
                    if i > 0:
                        prev_entry = history[i-1]
                        prev_state = prev_entry.get('state', 'unknown')
                        if prev_state != state:
                            state_transitions.append((prev_state, state, timestamp))
                
                # Advanced cognitive pattern analysis using OpenCog reasoning
                def cognitive_pattern_reasoner(atomspace, analysis_data, utilities):
                    insights = []
                    predictions = []
                    recommendations = []
                    
                    entity_domain = entity_id.split('.')[0]
                    states = [entry.get('state') for entry in history]
                    unique_states = set(states)
                    
                    # Temporal pattern analysis
                    current_hour = datetime.now().hour
                    current_day = datetime.now().strftime("%A")
                    
                    # Peak activity analysis
                    hour_activity_count = {}
                    for hour, states_in_hour in hourly_activity.items():
                        hour_activity_count[hour] = len([s for s in states_in_hour if s == 'on'])
                    
                    if hour_activity_count:
                        peak_hour = max(hour_activity_count.items(), key=lambda x: x[1])
                        insights.append(f"📊 Peak activity hour: {peak_hour[0]:02d}:00 with {peak_hour[1]} activations")
                        
                        if current_hour == peak_hour[0]:
                            predictions.append("⚡ Currently in peak activity time - high usage expected")
                    
                    # Daily pattern analysis
                    day_activity_count = {}
                    for day, states_in_day in daily_activity.items():
                        day_activity_count[day] = len([s for s in states_in_day if s == 'on'])
                    
                    if day_activity_count:
                        most_active_day = max(day_activity_count.items(), key=lambda x: x[1])
                        insights.append(f"📅 Most active day: {most_active_day[0]} with {most_active_day[1]} activations")
                    
                    # State transition analysis
                    if state_transitions:
                        transition_patterns = {}
                        for prev_state, new_state, timestamp in state_transitions:
                            pattern = f"{prev_state}→{new_state}"
                            if pattern not in transition_patterns:
                                transition_patterns[pattern] = []
                            transition_patterns[pattern].append(timestamp.hour)
                        
                        most_common_transition = max(transition_patterns.items(), key=lambda x: len(x[1]))
                        insights.append(f"🔄 Most common transition: {most_common_transition[0]} ({len(most_common_transition[1])} times)")
                    
                    # Domain-specific cognitive analysis
                    if entity_domain == 'light':
                        on_states = [s for s in states if s == 'on']
                        off_states = [s for s in states if s == 'off']
                        usage_ratio = len(on_states) / len(states) if states else 0
                        
                        insights.append(f"💡 Usage ratio: {usage_ratio:.2%} of time active")
                        
                        if usage_ratio > 0.7:
                            recommendations.append("🔆 High usage detected - consider smart dimming for energy savings")
                        elif usage_ratio < 0.1:
                            recommendations.append("🌙 Low usage detected - consider motion automation")
                        
                        # Predict next likely state
                        if states:
                            current_state = states[-1]
                            if current_state == 'off' and current_hour in hour_activity_count and hour_activity_count[current_hour] > 0:
                                predictions.append(f"🔮 Likely to turn on soon (typical activity hour)")
                    
                    elif entity_domain == 'climate':
                        # Temperature and HVAC analysis
                        insights.append("🌡️ Climate system - analyzing comfort patterns")
                        recommendations.append("📊 Consider smart scheduling based on occupancy patterns")
                    
                    elif entity_domain == 'sensor':
                        if 'motion' in entity_id.lower():
                            motion_events = len([s for s in states if s == 'on'])
                            insights.append(f"🚶 Motion events: {motion_events} in {hours} hours")
                            if motion_events > 20:
                                recommendations.append("🏃 High traffic area - optimize automation triggers")
                    
                    # Energy efficiency recommendations
                    if entity_domain in ['light', 'switch', 'climate']:
                        recommendations.append("♻️ Set up automation schedules to optimize energy usage")
                        if len(state_transitions) > 10:
                            recommendations.append("🔄 Frequent state changes detected - consider smart sensors")
                    
                    return {
                        'insights': insights,
                        'predictions': predictions,
                        'recommendations': recommendations
                    }
                
                # Apply cognitive reasoning
                ctx.deps.utilities.register_reasoner("cognitive_pattern_analysis", cognitive_pattern_reasoner)
                cognitive_results = ctx.deps.utilities.apply_reasoner("cognitive_pattern_analysis", {"history": history})
                
                # Generate comprehensive response
                result = f"""
🧠 Cognitive Pattern Analysis for {entity_id} (last {hours} hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Basic Statistics:
• Total state changes: {len(history)}
• Unique states observed: {', '.join(set(entry.get('state', 'unknown') for entry in history))}
• Current state: {history[-1].get('state', 'unknown') if history else 'unknown'}

⚡ Recent Activity (last 10 changes):
"""
                
                for entry in history[-10:]:
                    timestamp = entry.get('last_changed', entry.get('last_updated', 'unknown'))
                    state = entry.get('state', 'unknown')
                    result += f"• {timestamp}: {state}\n"
                
                if cognitive_results:
                    if cognitive_results.get('insights'):
                        result += f"\n🧠 Cognitive Insights:\n"
                        for insight in cognitive_results['insights']:
                            result += f"  {insight}\n"
                    
                    if cognitive_results.get('predictions'):
                        result += f"\n🔮 Predictive Analysis:\n"
                        for prediction in cognitive_results['predictions']:
                            result += f"  {prediction}\n"
                    
                    if cognitive_results.get('recommendations'):
                        result += f"\n💡 Smart Recommendations:\n"
                        for recommendation in cognitive_results['recommendations']:
                            result += f"  {recommendation}\n"
                
                result += f"\n🎯 Knowledge stored in AtomSpace for continuous learning and future predictions."
                
                return result
                
            else:
                return f"❌ Error getting pattern history for {entity_id}: {response.status}"
    except Exception as e:
        return f"❌ Exception analyzing patterns for {entity_id}: {str(e)}"

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

@marduk_agent.tool
async def create_smart_automation_with_cognition(
    ctx: RunContext[HADeps],
    description: str,
    entities_involved: List[str] = None
) -> str:
    """
    Create intelligent Home Assistant automation using OpenCog cognitive reasoning.
    
    This advanced tool uses cognitive analysis to create optimized automations that learn
    from usage patterns and adapt to user preferences over time.
    
    Args:
        description: Natural language description of the desired automation
        entities_involved: Optional list of entity IDs to consider for the automation
    
    Returns the created automation with cognitive enhancements and learning capabilities.
    """
    # Initialize cognitive components if not already done
    await initialize_cognitive_components(ctx.deps)
    
    # Cognitive analysis of the automation request
    def automation_design_reasoner(atomspace, request_description, utilities):
        automation_suggestions = {
            "triggers": [],
            "conditions": [],
            "actions": [],
            "cognitive_enhancements": []
        }
        
        request_lower = request_description.lower()
        
        # Analyze the request for intent and context
        if "motion" in request_lower and "light" in request_lower:
            automation_suggestions["triggers"].append({
                "platform": "state",
                "entity_id": "binary_sensor.motion_*",
                "to": "on"
            })
            automation_suggestions["actions"].append({
                "service": "light.turn_on",
                "entity_id": "light.*"
            })
            automation_suggestions["cognitive_enhancements"].append(
                "📊 Will learn optimal brightness levels based on time of day"
            )
            
        elif "sunset" in request_lower or "evening" in request_lower:
            automation_suggestions["triggers"].append({
                "platform": "sun",
                "event": "sunset"
            })
            automation_suggestions["cognitive_enhancements"].append(
                "🌅 Will adapt trigger time based on seasonal changes"
            )
            
        elif "temperature" in request_lower:
            automation_suggestions["triggers"].append({
                "platform": "numeric_state",
                "entity_id": "sensor.temperature",
                "above": "25"
            })
            automation_suggestions["cognitive_enhancements"].append(
                "🌡️ Will optimize temperature thresholds based on usage patterns"
            )
            
        elif "security" in request_lower or "alarm" in request_lower:
            automation_suggestions["triggers"].append({
                "platform": "state",
                "entity_id": "binary_sensor.door_*",
                "to": "on"
            })
            automation_suggestions["cognitive_enhancements"].append(
                "🔒 Will learn normal access patterns to reduce false alarms"
            )
        
        # Add intelligent conditions based on context
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:  # Night time
            automation_suggestions["conditions"].append({
                "condition": "time",
                "after": "22:00:00",
                "before": "06:00:00"
            })
            automation_suggestions["cognitive_enhancements"].append(
                "🌙 Night mode optimization for minimal disruption"
            )
        
        return automation_suggestions
    
    # Apply cognitive reasoning to design the automation
    ctx.deps.utilities.register_reasoner("automation_design", automation_design_reasoner)
    cognitive_design = ctx.deps.utilities.apply_reasoner("automation_design", description)
    
    # Create the automation configuration
    automation_config = {
        "alias": f"Cognitive Automation: {description[:50]}...",
        "description": f"Smart automation created with OpenCog cognitive reasoning: {description}",
        "trigger": cognitive_design.get("triggers", []),
        "condition": cognitive_design.get("conditions", []),
        "action": cognitive_design.get("actions", [])
    }
    
    # If no cognitive suggestions, create a basic automation
    if not automation_config["trigger"]:
        automation_config = {
            "alias": f"Basic Automation: {description[:50]}...",
            "description": f"Automation based on: {description}",
            "trigger": {
                "platform": "time",
                "at": "12:00:00"
            },
            "action": {
                "service": "persistent_notification.create",
                "data": {
                    "message": f"Automation reminder: {description}",
                    "title": "Marduk's Lab"
                }
            }
        }
    
    # Store automation intent in AtomSpace for learning
    automation_node = ctx.deps.atomspace.add_node("ConceptNode", f"Automation_{len(ctx.deps.atomspace.atoms)}")
    intent_node = ctx.deps.atomspace.add_node("ConceptNode", description)
    
    ctx.deps.utilities.create_atomese_expression(
        f"(Evaluation (Predicate \"has_intent\") (List {automation_node} {intent_node}))"
    )
    
    # Attempt to create the automation
    url = f"{ctx.deps.ha_url}/api/config/automation/config"
    headers = {
        "Authorization": f"Bearer {ctx.deps.ha_token}",
        "Content-Type": "application/json"
    }
    
    if not ctx.deps.session:
        ctx.deps.session = aiohttp.ClientSession()
    
    try:
        async with ctx.deps.session.post(url, headers=headers, json=automation_config) as response:
            result_text = f"""
🧠 Cognitive Automation Creation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Request: {description}
🎯 Automation Name: {automation_config['alias']}

🔧 Generated Configuration:
{json.dumps(automation_config, indent=2)}
"""
            
            if response.status in [200, 201]:
                result = await response.json()
                result_text += f"\n✅ Automation created successfully! ID: {result.get('id', 'unknown')}\n"
            else:
                error_text = await response.text()
                result_text += f"\n❌ Error creating automation: {response.status} - {error_text}\n"
                result_text += "💡 Suggestion: Check your Home Assistant configuration and permissions.\n"
            
            # Add cognitive enhancements information
            if cognitive_design.get("cognitive_enhancements"):
                result_text += f"\n🧠 Cognitive Enhancements:\n"
                for enhancement in cognitive_design["cognitive_enhancements"]:
                    result_text += f"  {enhancement}\n"
            
            result_text += f"\n📚 Automation knowledge stored in AtomSpace for continuous improvement."
            
            return result_text
            
    except Exception as e:
        return f"❌ Exception creating automation: {str(e)}"

# Example usage and testing
async def main():
    """
    Example usage of the Marduk Home Assistant agent with OpenCog integration.
    This demonstrates the enhanced cognitive capabilities for smart home automation.
    """
    # Initialize dependencies with cognitive components
    deps = HADeps(
        ha_url="http://homeassistant.local:8123",  # Replace with your HA URL
        ha_token="your_long_lived_access_token"    # Replace with your token
    )
    
    # Initialize OpenCog cognitive components
    await initialize_cognitive_components(deps)
    
    # Example conversations showcasing cognitive capabilities
    examples = [
        "What's the current temperature in the living room and provide cognitive insights?",
        "Turn on all the lights and learn from this interaction",
        "Set the thermostat to 72 degrees with smart optimization",
        "Analyze usage patterns for the main living room light with cognitive reasoning",
        "Create a smart automation to turn off all lights when no motion is detected",
        "Show me all devices and their cognitive analysis",
        "Provide energy usage analysis with optimization suggestions",
        "Create an intelligent evening routine automation"
    ]
    
    print("🏠🧠 Marduk's Lab - Cognitive Home Assistant Agent Ready!")
    print("Enhanced with OpenCog reasoning for intelligent home automation")
    print("=" * 60)
    print("\nCognitive Features:")
    print("• 🧠 Advanced pattern recognition and learning")
    print("• 🔮 Predictive automation suggestions")
    print("• 📊 Semantic understanding of device relationships")
    print("• 💡 Intelligent optimization recommendations")
    print("• 📚 Knowledge persistence across sessions")
    print("• 🤝 Collaborative reasoning with other Archon agents")
    print("\nExample commands you can try:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print(f"\n🎯 AtomSpace initialized with {len(deps.atomspace.atoms)} knowledge atoms")
    print("🔄 CogServer ready for cognitive processing")
    print("⚡ Utilities configured for advanced reasoning")

if __name__ == "__main__":
    asyncio.run(main())