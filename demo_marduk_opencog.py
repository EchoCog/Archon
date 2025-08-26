#!/usr/bin/env python3
"""
Marduk's Lab OpenCog Integration Demonstration
==============================================

This script showcases the enhanced cognitive capabilities that result from 
integrating OpenCog with Marduk's Lab home automation system.

Run this script to see how OpenCog provides advanced reasoning, pattern recognition,
and intelligent automation suggestions for smart home devices.
"""

import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.append('/home/runner/work/Archon/Archon')

def print_banner():
    """Print an attractive banner for the demonstration."""
    print("=" * 80)
    print("🏠🧠 MARDUK'S LAB - OPENCOG COGNITIVE INTEGRATION DEMO")
    print("=" * 80)
    print("Demonstrating intelligent home automation with cognitive reasoning")
    print("Powered by Archon + OpenCog + Home Assistant")
    print("=" * 80)
    print()

def demo_cognitive_home_automation():
    """Demonstrate the cognitive home automation capabilities."""
    
    print_banner()
    
    # Import OpenCog components
    try:
        from utils.opencog import opencog
        print("✅ OpenCog cognitive components loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load OpenCog: {e}")
        return False
    
    # Initialize the cognitive home automation system
    print("\n🔧 Initializing Cognitive Home Automation System...")
    print("-" * 60)
    
    atomspace = opencog.atomspace()
    cogserver = opencog.cogserver(atomspace)
    utilities = opencog.utilities(atomspace)
    
    print(f"🧠 AtomSpace initialized for knowledge storage")
    print(f"⚡ CogServer ready for cognitive processing")
    print(f"🔍 Utilities prepared for advanced reasoning")
    
    # Create home automation knowledge base
    print("\n🏠 Building Cognitive Knowledge Base...")
    print("-" * 60)
    
    # Core concepts
    home_node = atomspace.add_node("ConceptNode", "MarduksLab")
    automation_node = atomspace.add_node("ConceptNode", "CognitiveAutomation")
    user_node = atomspace.add_node("ConceptNode", "HomeOwner")
    
    # Device categories with relationships
    device_categories = {
        'lighting': ['light.living_room', 'light.kitchen', 'light.bedroom'],
        'climate': ['climate.main_floor', 'climate.upstairs'],
        'security': ['lock.front_door', 'lock.back_door', 'sensor.motion_kitchen'],
        'energy': ['switch.coffee_maker', 'switch.dishwasher', 'sensor.power_meter']
    }
    
    device_count = 0
    for category, devices in device_categories.items():
        category_node = atomspace.add_node("ConceptNode", f"{category.title()}System")
        utilities.create_atomese_expression(f"(Inheritance {category_node} {automation_node})")
        
        for device_id in devices:
            device_node = atomspace.add_node("ConceptNode", device_id)
            utilities.create_atomese_expression(f"(Inheritance {device_node} {category_node})")
            device_count += 1
        
        print(f"  ✅ {category.title()} system: {len(devices)} devices configured")
    
    print(f"\n📊 Knowledge base created with {len(atomspace.atoms)} cognitive atoms")
    print(f"🏠 Managing {device_count} smart home devices")
    
    # Simulate device interactions and learning
    print("\n⚡ Simulating Device Interactions and Learning...")
    print("-" * 60)
    
    # Create temporal patterns
    current_time = datetime.now()
    time_node = atomspace.add_node("TimeNode", current_time.isoformat())
    
    interactions = [
        ("light.living_room", "turn_on", "evening", "brightness=180"),
        ("climate.main_floor", "set_temperature", "evening", "temp=72"),
        ("lock.front_door", "lock", "night", "secure_mode=true"),
        ("switch.coffee_maker", "turn_on", "morning", "schedule=auto"),
        ("sensor.motion_kitchen", "detected", "night", "sensitivity=high")
    ]
    
    for device, action, time_context, params in interactions:
        # Store interaction in cognitive memory
        device_node = atomspace.add_node("ConceptNode", device)
        action_node = atomspace.add_node("ConceptNode", f"Action_{action}")
        context_node = atomspace.add_node("ConceptNode", time_context)
        
        utilities.create_atomese_expression(
            f"(AtTimeLink {time_node} "
            f"(Evaluation (Predicate \"{action}\") "
            f"(List {device_node} {action_node})))"
        )
        
        utilities.create_atomese_expression(
            f"(Evaluation (Predicate \"occurs_during\") "
            f"(List {action_node} {context_node}))"
        )
        
        print(f"  🎯 Learned: {device} -> {action} during {time_context} ({params})")
    
    # Demonstrate cognitive reasoning
    print("\n🧠 Applying Cognitive Reasoning for Smart Automation...")
    print("-" * 60)
    
    def marduk_cognitive_reasoner(atomspace, context, utilities=None):
        """Advanced cognitive reasoner for Marduk's Lab optimization."""
        
        insights = []
        automations = []
        optimizations = []
        
        # Analyze the knowledge base
        all_atoms = atomspace.atoms
        
        # Device analysis
        light_devices = [atom_id for atom_id in all_atoms.keys() if 'light.' in atom_id]
        climate_devices = [atom_id for atom_id in all_atoms.keys() if 'climate.' in atom_id]
        security_devices = [atom_id for atom_id in all_atoms.keys() if any(sec in atom_id for sec in ['lock.', 'sensor.'])]
        
        # Generate cognitive insights
        if light_devices:
            insights.append(f"💡 Lighting Analysis: {len(light_devices)} devices detected")
            automations.append("🌅 Morning Routine: Gradually increase brightness from 6-8 AM")
            automations.append("🌆 Evening Ambiance: Dim lights automatically after sunset")
            optimizations.append("⚡ Energy Saving: Motion-activated lighting in low-traffic areas")
        
        if climate_devices:
            insights.append(f"🌡️ Climate Analysis: {len(climate_devices)} HVAC zones identified") 
            automations.append("🏠 Occupancy Control: Adjust temperature based on room presence")
            automations.append("🌙 Night Setback: Lower temperature during sleeping hours")
            optimizations.append("📊 Energy Efficiency: Optimize schedules based on utility rates")
        
        if security_devices:
            insights.append(f"🔒 Security Analysis: {len(security_devices)} protective devices active")
            automations.append("🚨 Smart Alerts: Context-aware notifications for unusual activity")
            automations.append("🔐 Auto-Lock: Secure doors when everyone leaves")
            optimizations.append("👁️ Intelligent Monitoring: Adapt sensitivity based on time and occupancy")
        
        # Temporal reasoning
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 9:
            automations.append("☕ Morning Scene: Activate coffee maker, adjust lighting, set news briefing")
        elif 17 <= current_hour <= 21:
            automations.append("🏡 Evening Comfort: Optimize temperature, set mood lighting, prepare dinner mode")
        elif 22 <= current_hour or current_hour <= 6:
            automations.append("🌙 Night Mode: Security active, lights dimmed, climate optimized for sleep")
        
        # Cross-device coordination
        optimizations.append("🔄 Device Coordination: Synchronize lighting with climate for optimal comfort")
        optimizations.append("📱 Predictive Control: Learn patterns to anticipate user needs")
        optimizations.append("🌍 Weather Integration: Adjust automation based on weather conditions")
        
        return {
            'insights': insights,
            'automations': automations,
            'optimizations': optimizations
        }
    
    # Apply cognitive reasoning
    utilities.register_reasoner("marduk_optimizer", marduk_cognitive_reasoner)
    cognitive_results = utilities.apply_reasoner("marduk_optimizer", "optimize_home_automation")
    
    # Display results
    print("🎯 COGNITIVE INSIGHTS:")
    for insight in cognitive_results['insights']:
        print(f"  {insight}")
    
    print("\n🤖 INTELLIGENT AUTOMATION SUGGESTIONS:")
    for automation in cognitive_results['automations']:
        print(f"  {automation}")
    
    print("\n⚡ OPTIMIZATION OPPORTUNITIES:")
    for optimization in cognitive_results['optimizations']:
        print(f"  {optimization}")
    
    # Knowledge persistence demonstration  
    print(f"\n📚 Knowledge Base Status:")
    print("-" * 60)
    print(f"🧠 Total cognitive atoms: {len(atomspace.atoms)}")
    print(f"📊 Device relationships: {device_count} devices across 4 categories")
    print(f"⏰ Temporal patterns: {len(interactions)} interaction patterns learned")
    print(f"🔗 Reasoning rules: Active cognitive optimization engine")
    
    # Future capabilities preview
    print(f"\n🚀 Advanced Capabilities Enabled:")
    print("-" * 60)
    print("  🎯 Predictive automation based on learned patterns")
    print("  🧠 Semantic understanding of device relationships") 
    print("  📈 Continuous learning from user interactions")
    print("  🤝 Knowledge sharing between Archon agents")
    print("  ⚡ Real-time optimization suggestions")
    print("  🌐 Context-aware decision making")
    
    print("\n" + "=" * 80)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("🌟 Marduk's Lab enhanced with OpenCog cognitive capabilities")
    print("🏠 Your smart home is now ready for intelligent automation")
    print("=" * 80)
    
    return True

def main():
    """Main demonstration function."""
    try:
        success = demo_cognitive_home_automation()
        if success:
            print("\n✅ Integration demonstration successful!")
            print("🚀 Ready to deploy cognitive home automation with Marduk's Lab")
        else:
            print("\n❌ Demonstration encountered issues")
            
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()