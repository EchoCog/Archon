"""
CogServer implementation for OpenCog integration.

This module provides a custom implementation of the CogServer component
for managing cognitive processes as specified in PLAN.md.
"""

import asyncio
import threading
from typing import Dict, List, Any, Callable

class CogServer:
    """
    CogServer is OpenCog's server for managing cognitive processes and agents.
    It handles the execution of cognitive agents and provides communication channels.
    """
    
    def __init__(self, atomspace=None):
        """
        Initialize a new CogServer instance.
        
        Args:
            atomspace: The AtomSpace instance to associate with this CogServer
        """
        self.atomspace = atomspace
        self.agents = {}
        self.agent_threads = {}
        self.running = False
        self.event_queue = asyncio.Queue()
        self.event_handlers = {}
        self.default_cycle_duration = 0.1  # seconds
    
    def register_agent(self, name: str, agent_callable: Callable, priority: int = 0):
        """
        Register a cognitive agent with the CogServer.
        
        Args:
            name: The name of the agent
            agent_callable: The function or callable object that implements the agent
            priority: Priority level for execution (higher values = higher priority)
        """
        self.agents[name] = {
            "callable": agent_callable,
            "priority": priority,
            "active": False
        }
    
    def start_agent(self, name: str):
        """
        Start a registered agent.
        
        Args:
            name: The name of the agent to start
        """
        if name in self.agents and not self.agents[name]["active"]:
            self.agents[name]["active"] = True
            
            if self.running:
                # Start the agent thread
                self._start_agent_thread(name)
    
    def stop_agent(self, name: str):
        """
        Stop a running agent.
        
        Args:
            name: The name of the agent to stop
        """
        if name in self.agents:
            self.agents[name]["active"] = False
            
            if name in self.agent_threads:
                self.agent_threads[name].join()
                del self.agent_threads[name]
    
    def _start_agent_thread(self, name: str):
        """
        Start a thread for a specific agent.
        
        Args:
            name: The name of the agent
        """
        def agent_loop():
            agent_info = self.agents[name]
            while self.running and agent_info["active"]:
                try:
                    # Execute the agent
                    agent_info["callable"](self.atomspace)
                except Exception as e:
                    print(f"Error executing agent {name}: {e}")
                
                # Sleep according to priority (higher priority = shorter sleep)
                sleep_time = max(0.01, self.default_cycle_duration / (1 + agent_info["priority"] * 0.1))
                threading.Event().wait(sleep_time)
        
        thread = threading.Thread(target=agent_loop, name=f"agent-{name}")
        thread.daemon = True
        thread.start()
        self.agent_threads[name] = thread
    
    def start(self):
        """Start the CogServer."""
        if not self.running:
            self.running = True
            
            # Start all active agents
            for name, agent_info in self.agents.items():
                if agent_info["active"]:
                    self._start_agent_thread(name)
    
    def stop(self):
        """Stop the CogServer."""
        self.running = False
        
        # Join all agent threads
        for name, thread in list(self.agent_threads.items()):
            thread.join()
            del self.agent_threads[name]
    
    async def publish_event(self, event_type: str, data: Any):
        """
        Publish an event to subscribers.
        
        Args:
            event_type: The type of event
            data: The event data
        """
        await self.event_queue.put((event_type, data))
    
    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to an event type.
        
        Args:
            event_type: The type of event to subscribe to
            callback: The callback function to call when the event occurs
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(callback)
    
    async def event_loop(self):
        """Process events from the event queue."""
        while self.running:
            try:
                event_type, data = await self.event_queue.get()
                
                # Notify subscribers
                if event_type in self.event_handlers:
                    for callback in self.event_handlers[event_type]:
                        try:
                            callback(data)
                        except Exception as e:
                            print(f"Error in event handler for {event_type}: {e}")
                
                self.event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in event loop: {e}")

    def get_agent_status(self) -> Dict[str, Dict]:
        """
        Get the status of all registered agents.
        
        Returns:
            Dictionary with agent status information
        """
        status = {}
        for name, agent_info in self.agents.items():
            status[name] = {
                "active": agent_info["active"],
                "priority": agent_info["priority"],
                "running": name in self.agent_threads and self.agent_threads[name].is_alive()
            }
        return status
