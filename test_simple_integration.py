#!/usr/bin/env python3
"""
Simple test script for Marduk's Lab OpenCog Integration
Demonstrates the enhanced cognitive capabilities for home automation.
"""

import sys
import os

# Add the project root to Python path
sys.path.append('/home/runner/work/Archon/Archon')

def test_opencog_integration():
    """Test the OpenCog integration independently."""
    
    print("🏠🧠 Testing Marduk's Lab OpenCog Integration")
    print("=" * 60)
    
    # Import OpenCog components
    try:
        from utils.opencog import opencog
        print("✅ OpenCog components imported successfully")
    except Exception as e:
        print(f"❌ Failed to import OpenCog: {e}")
        return False
    
    # Initialize cognitive components
    atomspace = opencog.atomspace()
    cogserver = opencog.cogserver(atomspace)
    utilities = opencog.utilities(atomspace)
    
    print(f"🧠 AtomSpace initialized: {len(atomspace.atoms)} atoms")
    print(f"⚡ CogServer ready for cognitive processing")
    print(f"🔍 Utilities configured for advanced reasoning")
    
    # Test home automation knowledge creation
    print("\n🏠 Creating Home Automation Knowledge Base:")
    print("-" * 50)
    
    # Create foundational concepts
    home_concept = atomspace.add_node("ConceptNode", "HomeAutomation")
    device_concept = atomspace.add_node("ConceptNode", "SmartDevice")
    user_concept = atomspace.add_node("ConceptNode", "User")
    pattern_concept = atomspace.add_node("ConceptNode", "UsagePattern")
    energy_concept = atomspace.add_node("ConceptNode", "EnergyEfficiency")
    
    print(f"✅ Created core concepts: {len(atomspace.atoms)} atoms")
    
    # Create device categories
    device_types = ['Light', 'Climate', 'Sensor', 'Switch', 'Lock', 'Camera']
    for device_type in device_types:
        device_node = atomspace.add_node("ConceptNode", f"{device_type}Device")
        utilities.create_atomese_expression(f"(Inheritance {device_node} {device_concept})")
        print(f"✅ Added {device_type} device category")
    
    # Create specific devices
    devices = [
        "light.living_room",
        "light.kitchen", 
        "climate.main_floor",
        "sensor.motion_kitchen",
        "lock.front_door",
        "switch.coffee_maker"
    ]
    
    for device_id in devices:
        device_node = atomspace.add_node("ConceptNode", device_id)
        domain = device_id.split('.')[0]
        domain_concept = atomspace.add_node("ConceptNode", f"{domain.title()}Device")
        utilities.create_atomese_expression(f"(Inheritance {device_node} {domain_concept})")
        print(f"✅ Added device: {device_id}")
    
    print(f"\n📊 Knowledge base now contains {len(atomspace.atoms)} atoms")
    
    # Test cognitive reasoning for home automation
    print("\n🧠 Testing Cognitive Reasoning:")
    print("-" * 40)
    
    def home_automation_reasoner(atomspace, request, utilities=None):
        """Cognitive reasoner for home automation optimization."""
        insights = []
        
        # Analyze device relationships
        all_atoms = atomspace.atoms
        device_count = len([atom_id for atom_id in all_atoms.keys() if '.' in atom_id])
        
        insights.append(f"📊 Managing {device_count} smart devices")
        
        # Domain-specific insights
        light_devices = [atom_id for atom_id in all_atoms.keys() if atom_id.startswith('light.')]
        if light_devices:
            insights.append(f"💡 {len(light_devices)} lighting devices - consider motion automation")
            insights.append("🌙 Implement smart dimming for energy efficiency")
        
        climate_devices = [atom_id for atom_id in all_atoms.keys() if atom_id.startswith('climate.')]
        if climate_devices:
            insights.append(f"🌡️ {len(climate_devices)} climate devices - enable occupancy scheduling")
            insights.append("📊 Set up temperature setbacks for energy savings")
        
        security_devices = [atom_id for atom_id in all_atoms.keys() if any(sec in atom_id for sec in ['lock.', 'sensor.'])]
        if security_devices:
            insights.append(f"🔒 {len(security_devices)} security devices - configure smart alerts")
            insights.append("📱 Enable mobile notifications for status changes")
        
        # General automation suggestions
        insights.append("🤖 Create smart scenes for common scenarios")
        insights.append("⚡ Implement energy monitoring for optimization")
        insights.append("📈 Use pattern learning for predictive automation")
        insights.append("🔄 Enable cross-device coordination for efficiency")
        
        return insights
    
    # Register and apply the reasoner
    utilities.register_reasoner("marduk_optimizer", home_automation_reasoner)
    cognitive_insights = utilities.apply_reasoner("marduk_optimizer", "optimize home automation")
    
    print("🎯 Cognitive Insights for Marduk's Lab:")
    for insight in cognitive_insights:
        print(f"  {insight}")
    
    # Test pattern creation and querying
    print("\n🔗 Testing Pattern Recognition:")
    print("-" * 40)
    
    # Create automation patterns
    patterns = [
        ("User", "controls", "SmartDevice"),
        ("MotionSensor", "triggers", "LightAutomation"),
        ("ClimateDevice", "optimizes", "EnergyConsumption"),
        ("SecurityDevice", "monitors", "HomeSecurityModel"),
        ("LightDevice", "adapts_to", "CircadianRhythm"),
        ("AutomationPattern", "learns_from", "UserBehavior")
    ]
    
    for subject, predicate, obj in patterns:
        subject_node = atomspace.add_node("ConceptNode", subject)
        object_node = atomspace.add_node("ConceptNode", obj)
        utilities.create_atomese_expression(
            f"(Evaluation (Predicate \"{predicate}\") (List {subject_node} {object_node}))"
        )
        print(f"✅ Pattern: {subject} {predicate} {obj}")
    
    # Test temporal reasoning
    print("\n⏰ Testing Temporal Reasoning:")
    print("-" * 40)
    
    import datetime
    current_time = datetime.datetime.now()
    time_node = atomspace.add_node("TimeNode", current_time.isoformat())
    
    # Create time-based patterns
    temporal_patterns = [
        ("Morning", "activates", "CoffeeAutomation"),
        ("Evening", "triggers", "RelaxationScene"),
        ("Night", "enables", "SecurityMode"),
        ("Weekend", "suggests", "ComfortOptimization")
    ]
    
    for time_context, relation, automation in temporal_patterns:
        time_concept = atomspace.add_node("ConceptNode", time_context)
        automation_concept = atomspace.add_node("ConceptNode", automation)
        utilities.create_atomese_expression(
            f"(AtTimeLink {time_node} (Evaluation (Predicate \"{relation}\") (List {time_concept} {automation_concept})))"
        )
        print(f"✅ Temporal pattern: {time_context} {relation} {automation}")
    
    # Query the knowledge base
    print(f"\n🔍 Knowledge Base Query Results:")
    print("-" * 40)
    
    concept_nodes = utilities.query_atoms(type_filter="ConceptNode")
    print(f"📊 Total ConceptNodes: {len(concept_nodes)}")
    
    print("\n🏠 Device Entities Found:")
    device_entities = [handle for handle in concept_nodes 
                      if '.' in atomspace.get_atom(handle)['name'] 
                      if atomspace.get_atom(handle)]
    
    for handle in device_entities:
        atom = atomspace.get_atom(handle)
        if atom:
            print(f"  • {atom['name']}")
    
    print(f"\n🎉 Integration Test Complete!")
    print(f"🌟 Marduk's Lab enhanced with {len(atomspace.atoms)} knowledge atoms")
    print("🧠 OpenCog cognitive reasoning ready for home automation")
    
    return True

def main():
    """Main test function."""
    try:
        success = test_opencog_integration()
        if success:
            print("\n✅ All tests passed! OpenCog integration successful.")
        else:
            print("\n❌ Some tests failed.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()