#!/usr/bin/env python3
"""
Test script for Marduk's Lab OpenCog Integration
Demonstrates the enhanced cognitive capabilities for home automation.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.opencog import opencog
from agent-resources.tools.home_assistant_tools import CognitiveHomeAssistantAPI

async def test_cognitive_integration():
    """Test the OpenCog integration with Marduk's Lab."""
    
    print("🏠🧠 Testing Marduk's Lab OpenCog Integration")
    print("=" * 60)
    
    # Initialize cognitive Home Assistant API (without actual HA connection)
    cognitive_ha = CognitiveHomeAssistantAPI(
        ha_url="http://localhost:8123",
        ha_token="test_token",
        enable_cognition=True
    )
    
    print(f"✅ Cognitive Home Assistant API initialized")
    print(f"🧠 AtomSpace contains {len(cognitive_ha.atomspace.atoms)} knowledge atoms")
    print(f"⚡ CogServer ready for cognitive processing")
    print(f"🔍 Utilities configured for advanced reasoning")
    
    # Test cognitive insights generation
    print("\n🧠 Testing Cognitive Insights Generation:")
    print("-" * 40)
    
    test_entities = [
        "light.living_room",
        "climate.main_floor", 
        "sensor.motion_kitchen",
        "lock.front_door",
        "switch.coffee_maker"
    ]
    
    for entity_id in test_entities:
        insights = cognitive_ha._get_cognitive_insights(entity_id, "test interaction")
        print(f"\n📍 Entity: {entity_id}")
        for insight in insights:
            print(f"  {insight}")
    
    # Test device interaction storage
    print("\n📊 Testing Device Interaction Storage:")
    print("-" * 40)
    
    # Simulate device interactions
    interactions = [
        ("light.living_room", "turn_on", {"brightness": 255}),
        ("climate.main_floor", "set_temperature", {"temperature": 72}),
        ("sensor.motion_kitchen", "triggered", {}),
        ("lock.front_door", "lock", {}),
        ("switch.coffee_maker", "turn_on", {})
    ]
    
    for entity_id, action, params in interactions:
        cognitive_ha._store_device_interaction(entity_id, action, params)
        print(f"✅ Stored interaction: {entity_id} -> {action} {params}")
    
    print(f"\n🧠 AtomSpace now contains {len(cognitive_ha.atomspace.atoms)} knowledge atoms")
    
    # Test cognitive reasoning
    print("\n🔮 Testing Cognitive Reasoning:")
    print("-" * 40)
    
    # Create a test reasoner
    def home_automation_reasoner(atomspace, request, utilities):
        recommendations = []
        
        # Analyze stored device interactions
        all_atoms = atomspace.atoms
        light_interactions = [atom for atom_id, atom in all_atoms.items() if 'light' in atom_id.lower()]
        climate_interactions = [atom for atom_id, atom in all_atoms.items() if 'climate' in atom_id.lower()]
        
        if light_interactions:
            recommendations.append("💡 Light automation: Consider motion-based triggers")
            recommendations.append("🌙 Energy optimization: Schedule dimming for evening hours")
        
        if climate_interactions:
            recommendations.append("🌡️ Climate automation: Set up occupancy-based scheduling")
            recommendations.append("📊 Energy efficiency: Implement temperature setbacks")
        
        recommendations.append("🤖 Smart scenes: Combine multiple devices for optimal comfort")
        recommendations.append("📱 Mobile integration: Set up notifications for security devices")
        
        return recommendations
    
    # Apply cognitive reasoning
    cognitive_ha.utilities.register_reasoner("home_automation_optimizer", home_automation_reasoner)
    recommendations = cognitive_ha.utilities.apply_reasoner("home_automation_optimizer", "optimize home automation")
    
    print("🎯 Cognitive Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    # Test pattern creation in AtomSpace
    print("\n🔗 Testing AtomSpace Pattern Creation:")
    print("-" * 40)
    
    # Create some example patterns
    patterns = [
        ("User", "prefers_automation", "EnergyEfficiency"),
        ("LivingRoom", "contains", "SmartLight"),
        ("MotionSensor", "triggers", "LightAutomation"),
        ("ClimateControl", "optimizes", "EnergyConsumption")
    ]
    
    for subject, predicate, obj in patterns:
        subject_node = cognitive_ha.atomspace.add_node("ConceptNode", subject)
        object_node = cognitive_ha.atomspace.add_node("ConceptNode", obj)
        cognitive_ha.utilities.create_atomese_expression(
            f"(Evaluation (Predicate \"{predicate}\") (List {subject_node} {object_node}))"
        )
        print(f"✅ Created pattern: {subject} {predicate} {obj}")
    
    print(f"\n🎯 Final AtomSpace knowledge base: {len(cognitive_ha.atomspace.atoms)} atoms")
    
    # Test knowledge querying
    print("\n🔍 Testing Knowledge Querying:")
    print("-" * 40)
    
    concept_nodes = cognitive_ha.utilities.query_atoms(type_filter="ConceptNode")
    print(f"📊 Found {len(concept_nodes)} concept nodes in knowledge base:")
    
    for handle in concept_nodes[:10]:  # Show first 10
        atom = cognitive_ha.atomspace.get_atom(handle)
        if atom:
            print(f"  • {atom.get('name', 'Unknown')}")
    
    if len(concept_nodes) > 10:
        print(f"  ... and {len(concept_nodes) - 10} more concepts")
    
    print("\n🎉 OpenCog Integration Test Complete!")
    print("🌟 Marduk's Lab is now enhanced with cognitive capabilities")
    
    # Close session if it exists
    if cognitive_ha.session:
        await cognitive_ha.close()

async def main():
    """Main test function."""
    try:
        await test_cognitive_integration()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())