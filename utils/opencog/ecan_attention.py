"""
ECAN Attention Allocation & Resource Kernel Implementation

This module implements the Economic Cognitive Agent Network (ECAN) attention allocation
system as part of Phase 2 of the Archon cognitive architecture.

ECAN provides dynamic, economic-style attention allocation using:
- Short-Term Importance (STI) values for immediate attention
- Long-Term Importance (LTI) values for persistent relevance
- Activation spreading for attention propagation
- Economic resource allocation based on attention values

Tensor signature mapping: [tasks, attention, priority, resources]
- tasks → modality (type of cognitive task)
- attention → salience (attention weight, existing)
- priority → depth (processing priority level)
- resources → autonomy_index (resource allocation autonomy)
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import threading


class AttentionType(Enum):
    """Types of attention values in ECAN."""
    STI = "short_term_importance"  # Immediate attention focus
    LTI = "long_term_importance"   # Persistent relevance
    VLTI = "very_long_term_importance"  # Ultra-persistent significance


@dataclass
class AttentionValue:
    """
    Represents an attention value with STI, LTI, and confidence.
    
    This maps to the tensor signature as:
    - STI → salience (immediate attention)
    - LTI → context (long-term relevance)
    - confidence → autonomy_index (certainty of allocation)
    """
    sti: float = 0.0  # Short-term importance (-1000 to +1000)
    lti: float = 0.0  # Long-term importance (0 to +1000)
    vlti: float = 0.0  # Very long-term importance (0 to +1000)
    confidence: float = 1.0  # Confidence in attention allocation (0 to 1)
    
    def normalize_for_tensor(self) -> Tuple[float, float, float]:
        """
        Normalize attention values to [0,1] range for tensor fragment integration.
        
        Returns:
            Tuple of (normalized_sti, normalized_lti, normalized_confidence)
        """
        # Normalize STI from [-1000, 1000] to [0, 1]
        norm_sti = (self.sti + 1000) / 2000
        # Normalize LTI from [0, 1000] to [0, 1]  
        norm_lti = self.lti / 1000
        # Confidence is already in [0, 1]
        return (np.clip(norm_sti, 0, 1), np.clip(norm_lti, 0, 1), np.clip(self.confidence, 0, 1))
    
    @classmethod
    def from_tensor_values(cls, norm_sti: float, norm_lti: float, confidence: float) -> 'AttentionValue':
        """
        Create AttentionValue from normalized tensor values.
        
        Args:
            norm_sti: Normalized STI in [0, 1]
            norm_lti: Normalized LTI in [0, 1]
            confidence: Confidence in [0, 1]
        """
        # Denormalize STI from [0, 1] to [-1000, 1000]
        sti = (norm_sti * 2000) - 1000
        # Denormalize LTI from [0, 1] to [0, 1000]
        lti = norm_lti * 1000
        return cls(sti=sti, lti=lti, confidence=confidence)


@dataclass
class ResourceAllocation:
    """
    Represents resource allocation for a cognitive task.
    
    Maps to tensor signature as:
    - cpu_cycles → depth (processing complexity)
    - memory_allocation → modality (resource type)
    - priority_level → salience (attention priority)
    """
    cpu_cycles: int = 0
    memory_allocation: int = 0  # in MB
    priority_level: int = 0  # 0-10 scale
    task_type: str = "general"
    allocated_time: float = 0.0  # seconds
    
    def to_tensor_signature(self) -> Tuple[float, float, float, float]:
        """
        Convert resource allocation to tensor signature components.
        
        Returns:
            Tuple of (modality, depth, salience, autonomy) normalized to [0,1]
        """
        # Map task_type to modality value
        task_type_map = {
            "general": 0.1, "reasoning": 0.3, "perception": 0.5, 
            "memory": 0.7, "action": 0.9
        }
        modality = task_type_map.get(self.task_type, 0.5)
        
        # Normalize depth from processing complexity
        depth = min(self.cpu_cycles / 1000.0, 1.0)
        
        # Normalize salience from priority
        salience = self.priority_level / 10.0
        
        # Normalize autonomy from memory allocation
        autonomy = min(self.memory_allocation / 1000.0, 1.0)
        
        return (modality, depth, salience, autonomy)


class ECANAttentionManager:
    """
    Main ECAN attention allocation manager.
    
    Implements economic-style attention allocation with:
    - Dynamic STI/LTI management
    - Attention spreading and activation propagation
    - Resource scheduling based on attention values
    - Integration with existing tensor fragment architecture
    """
    
    def __init__(self, atomspace=None, initial_sti_budget: float = 10000.0, 
                 initial_lti_budget: float = 10000.0):
        """
        Initialize the ECAN attention manager.
        
        Args:
            atomspace: AtomSpace instance for integration
            initial_sti_budget: Total STI budget for allocation
            initial_lti_budget: Total LTI budget for allocation
        """
        self.atomspace = atomspace
        self.sti_budget = initial_sti_budget
        self.lti_budget = initial_lti_budget
        
        # Attention value storage
        self.attention_values: Dict[str, AttentionValue] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
        # Priority queues for scheduling
        self.sti_priority_queue = []  # Max heap for high STI atoms
        self.task_priority_queue = []  # Min heap for scheduled tasks
        
        # Attention spreading parameters
        self.spreading_factor = 0.8  # Attention decay factor
        self.min_sti_threshold = 1.0  # Minimum STI for processing
        self.activation_spread_rate = 0.1  # Rate of attention spreading
        
        # Resource management
        self.total_cpu_budget = 1000  # Total CPU cycles available
        self.total_memory_budget = 1000  # Total memory in MB
        self.allocated_cpu = 0
        self.allocated_memory = 0
        
        # Event handling
        self.attention_update_callbacks: List[Callable] = []
        self.resource_update_callbacks: List[Callable] = []
        
        # Threading for async operations
        self.running = False
        self.attention_thread = None
        self.lock = threading.RLock()
    
    def set_attention_value(self, atom_handle: str, attention_value: AttentionValue) -> bool:
        """
        Set attention value for an atom.
        
        Args:
            atom_handle: Handle of the atom
            attention_value: New attention value
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Check budget constraints
            current_av = self.attention_values.get(atom_handle, AttentionValue())
            sti_delta = attention_value.sti - current_av.sti
            lti_delta = attention_value.lti - current_av.lti
            
            # Enforce budget constraints
            if abs(sti_delta) > self.sti_budget or abs(lti_delta) > self.lti_budget:
                return False
            
            # Update attention value
            self.attention_values[atom_handle] = attention_value
            
            # Update priority queue if STI changed significantly
            if abs(sti_delta) > 1.0:
                self._update_priority_queue(atom_handle, attention_value.sti)
            
            # Update tensor fragment if atomspace is available
            if self.atomspace:
                self._sync_with_tensor_fragment(atom_handle, attention_value)
            
            # Trigger callbacks
            for callback in self.attention_update_callbacks:
                try:
                    callback(atom_handle, attention_value)
                except Exception as e:
                    print(f"Attention callback error: {e}")
            
            return True
    
    def get_attention_value(self, atom_handle: str) -> Optional[AttentionValue]:
        """Get attention value for an atom."""
        with self.lock:
            return self.attention_values.get(atom_handle)
    
    def allocate_resources(self, task_id: str, resource_req: ResourceAllocation) -> bool:
        """
        Allocate resources for a cognitive task.
        
        Args:
            task_id: Unique identifier for the task
            resource_req: Resource requirements
            
        Returns:
            True if resources allocated successfully
        """
        with self.lock:
            # Check resource availability
            if (self.allocated_cpu + resource_req.cpu_cycles > self.total_cpu_budget or
                self.allocated_memory + resource_req.memory_allocation > self.total_memory_budget):
                return False
            
            # Allocate resources
            self.resource_allocations[task_id] = resource_req
            self.allocated_cpu += resource_req.cpu_cycles
            self.allocated_memory += resource_req.memory_allocation
            
            # Add to task queue with priority
            priority_score = self._calculate_task_priority(task_id, resource_req)
            heapq.heappush(self.task_priority_queue, (priority_score, time.time(), task_id))
            
            # Trigger callbacks
            for callback in self.resource_update_callbacks:
                try:
                    callback(task_id, resource_req)
                except Exception as e:
                    print(f"Resource callback error: {e}")
            
            return True
    
    def deallocate_resources(self, task_id: str) -> bool:
        """
        Deallocate resources for a completed task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if successfully deallocated
        """
        with self.lock:
            if task_id not in self.resource_allocations:
                return False
            
            resource_req = self.resource_allocations[task_id]
            self.allocated_cpu -= resource_req.cpu_cycles
            self.allocated_memory -= resource_req.memory_allocation
            
            del self.resource_allocations[task_id]
            return True
    
    def spread_attention(self, source_atom: str, spread_amount: float = None) -> int:
        """
        Spread attention from source atom to connected atoms.
        
        Args:
            source_atom: Source atom handle
            spread_amount: Amount of attention to spread (default: auto-calculate)
            
        Returns:
            Number of atoms affected by spreading
        """
        if not self.atomspace or source_atom not in self.attention_values:
            return 0
        
        source_av = self.attention_values[source_atom]
        if spread_amount is None:
            spread_amount = source_av.sti * self.activation_spread_rate
        
        if spread_amount <= 0:
            return 0
        
        # Find connected atoms through atomspace relationships
        connected_atoms = self._find_connected_atoms(source_atom)
        if not connected_atoms:
            return 0
        
        # Distribute attention among connected atoms
        attention_per_atom = spread_amount / len(connected_atoms)
        affected_count = 0
        
        with self.lock:
            for connected_atom in connected_atoms:
                if connected_atom == source_atom:
                    continue
                
                # Get or create attention value for connected atom
                target_av = self.attention_values.get(connected_atom, AttentionValue())
                
                # Apply spreading with decay
                spread_sti = attention_per_atom * self.spreading_factor
                new_sti = target_av.sti + spread_sti
                
                # Update attention value
                new_av = AttentionValue(
                    sti=new_sti,
                    lti=target_av.lti,
                    vlti=target_av.vlti,
                    confidence=target_av.confidence
                )
                
                if self.set_attention_value(connected_atom, new_av):
                    affected_count += 1
        
        return affected_count
    
    def get_high_sti_atoms(self, count: int = 10) -> List[Tuple[str, float]]:
        """
        Get atoms with highest STI values.
        
        Args:
            count: Number of atoms to return
            
        Returns:
            List of (atom_handle, sti_value) tuples
        """
        with self.lock:
            # Sort by STI value in descending order
            sorted_atoms = sorted(
                self.attention_values.items(),
                key=lambda x: x[1].sti,
                reverse=True
            )
            return [(handle, av.sti) for handle, av in sorted_atoms[:count]]
    
    def get_next_scheduled_task(self) -> Optional[Tuple[str, ResourceAllocation]]:
        """
        Get the next task from the priority queue.
        
        Returns:
            Tuple of (task_id, resource_allocation) or None if queue empty
        """
        with self.lock:
            if not self.task_priority_queue:
                return None
            
            _, _, task_id = heapq.heappop(self.task_priority_queue)
            resource_alloc = self.resource_allocations.get(task_id)
            
            if resource_alloc:
                return (task_id, resource_alloc)
            return None
    
    def start_attention_daemon(self):
        """Start the background attention management daemon."""
        if self.running:
            return
        
        self.running = True
        self.attention_thread = threading.Thread(target=self._attention_daemon_loop)
        self.attention_thread.daemon = True
        self.attention_thread.start()
    
    def stop_attention_daemon(self):
        """Stop the background attention management daemon."""
        self.running = False
        if self.attention_thread:
            self.attention_thread.join()
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get current attention allocation statistics."""
        with self.lock:
            total_sti = sum(av.sti for av in self.attention_values.values())
            total_lti = sum(av.lti for av in self.attention_values.values())
            
            return {
                'total_atoms': len(self.attention_values),
                'total_sti': total_sti,
                'total_lti': total_lti,
                'sti_budget_used': self.sti_budget - abs(total_sti),
                'lti_budget_used': self.lti_budget - abs(total_lti),
                'cpu_utilization': self.allocated_cpu / self.total_cpu_budget,
                'memory_utilization': self.allocated_memory / self.total_memory_budget,
                'active_tasks': len(self.resource_allocations),
                'queued_tasks': len(self.task_priority_queue)
            }
    
    # Private helper methods
    
    def _update_priority_queue(self, atom_handle: str, sti_value: float):
        """Update the STI priority queue."""
        # Use negative STI for max heap behavior with heapq (min heap)
        heapq.heappush(self.sti_priority_queue, (-sti_value, time.time(), atom_handle))
    
    def _sync_with_tensor_fragment(self, atom_handle: str, attention_value: AttentionValue):
        """Synchronize attention values with tensor fragment salience."""
        if not self.atomspace:
            return
        
        # Get normalized attention values
        norm_sti, norm_lti, confidence = attention_value.normalize_for_tensor()
        
        # Update corresponding tensor fragment if it exists
        atom = self.atomspace.get_atom(atom_handle)
        if atom and 'tensor_fragment' in atom:
            # Update salience in the tensor signature
            atom['tensor_fragment']['signature']['salience'] = norm_sti
            atom['tensor_fragment']['signature']['context'] = norm_lti
            atom['tensor_fragment']['signature']['autonomy_index'] = confidence
    
    def _find_connected_atoms(self, atom_handle: str) -> List[str]:
        """Find atoms connected to the given atom."""
        if not self.atomspace:
            return []
        
        connected = []
        
        # Check incoming links
        for link_handle, link_data in self.atomspace.relationships.items():
            if atom_handle in link_data.get('outgoing_set', []):
                connected.extend(link_data['outgoing_set'])
        
        # Check outgoing links  
        atom = self.atomspace.get_atom(atom_handle)
        if atom and 'outgoing_set' in atom:
            connected.extend(atom['outgoing_set'])
        
        return list(set(connected))  # Remove duplicates
    
    def _calculate_task_priority(self, task_id: str, resource_req: ResourceAllocation) -> float:
        """Calculate priority score for task scheduling."""
        # Higher priority = lower score (for min heap)
        base_priority = 10 - resource_req.priority_level
        
        # Factor in resource intensity
        resource_factor = (resource_req.cpu_cycles / 100.0) + (resource_req.memory_allocation / 100.0)
        
        # Factor in attention value if available
        attention_factor = 0
        if task_id in self.attention_values:
            av = self.attention_values[task_id]
            attention_factor = -(av.sti / 100.0)  # Higher STI = lower score
        
        return base_priority + resource_factor + attention_factor
    
    def _attention_daemon_loop(self):
        """Background daemon for attention management."""
        while self.running:
            try:
                # Perform attention decay
                self._decay_attention_values()
                
                # Perform automatic attention spreading
                self._automatic_attention_spreading()
                
                # Cleanup low-attention atoms
                self._cleanup_low_attention_atoms()
                
                # Sleep for next cycle
                time.sleep(1.0)  # 1 second cycle
                
            except Exception as e:
                print(f"Attention daemon error: {e}")
    
    def _decay_attention_values(self):
        """Apply time-based decay to attention values."""
        decay_rate = 0.99  # 1% decay per cycle
        
        with self.lock:
            for atom_handle, av in list(self.attention_values.items()):
                # Decay STI toward zero
                new_sti = av.sti * decay_rate
                if abs(new_sti) < 0.1:
                    new_sti = 0.0
                
                # Update attention value
                new_av = AttentionValue(
                    sti=new_sti,
                    lti=av.lti,
                    vlti=av.vlti,
                    confidence=av.confidence * 0.999  # Slight confidence decay
                )
                
                self.attention_values[atom_handle] = new_av
    
    def _automatic_attention_spreading(self):
        """Automatically spread attention from high-STI atoms."""
        high_sti_atoms = self.get_high_sti_atoms(5)  # Top 5 atoms
        
        for atom_handle, sti_value in high_sti_atoms:
            if sti_value > 10.0:  # Only spread from significantly active atoms
                self.spread_attention(atom_handle)
    
    def _cleanup_low_attention_atoms(self):
        """Remove atoms with very low attention values to free memory."""
        with self.lock:
            to_remove = []
            for atom_handle, av in self.attention_values.items():
                if abs(av.sti) < 0.1 and av.lti < 0.1 and av.vlti < 0.1:
                    to_remove.append(atom_handle)
            
            for atom_handle in to_remove:
                del self.attention_values[atom_handle]


class ECANResourceKernel:
    """
    Resource kernel for ECAN-based cognitive task scheduling.
    
    This component manages the allocation and scheduling of computational resources
    based on attention values and cognitive task priorities.
    """
    
    def __init__(self, attention_manager: ECANAttentionManager):
        """
        Initialize the resource kernel.
        
        Args:
            attention_manager: ECAN attention manager instance
        """
        self.attention_manager = attention_manager
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Resource scheduling parameters
        self.max_concurrent_tasks = 10
        self.task_timeout = 30.0  # seconds
        self.scheduling_algorithm = "attention_weighted"
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.average_completion_time = 0.0
        self.resource_efficiency = 0.0
    
    async def schedule_cognitive_task(self, task_id: str, task_callable: Callable, 
                                    resource_req: ResourceAllocation,
                                    attention_boost: float = 0.0) -> bool:
        """
        Schedule a cognitive task for execution.
        
        Args:
            task_id: Unique task identifier
            task_callable: Function to execute for the task
            resource_req: Resource requirements
            attention_boost: Additional attention boost for this task
        
        Returns:
            True if task scheduled successfully
        """
        # Check if we can allocate resources
        if not self.attention_manager.allocate_resources(task_id, resource_req):
            return False
        
        # Apply attention boost if specified
        if attention_boost > 0.0:
            current_av = self.attention_manager.get_attention_value(task_id) or AttentionValue()
            boosted_av = AttentionValue(
                sti=current_av.sti + attention_boost,
                lti=current_av.lti,
                vlti=current_av.vlti,
                confidence=current_av.confidence
            )
            self.attention_manager.set_attention_value(task_id, boosted_av)
        
        # Store task information
        task_info = {
            'task_id': task_id,
            'callable': task_callable,
            'resource_req': resource_req,
            'start_time': time.time(),
            'status': 'scheduled'
        }
        
        self.active_tasks[task_id] = task_info
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_task(task_info))
        
        return True
    
    async def _execute_task(self, task_info: Dict[str, Any]):
        """Execute a cognitive task."""
        task_id = task_info['task_id']
        
        try:
            # Update status
            task_info['status'] = 'running'
            task_info['execution_start'] = time.time()
            
            # Execute the task
            result = await asyncio.wait_for(
                self._run_task_callable(task_info['callable']),
                timeout=self.task_timeout
            )
            
            # Task completed successfully
            task_info['status'] = 'completed'
            task_info['result'] = result
            task_info['completion_time'] = time.time()
            
        except asyncio.TimeoutError:
            task_info['status'] = 'timeout'
            task_info['completion_time'] = time.time()
            
        except Exception as e:
            task_info['status'] = 'error'
            task_info['error'] = str(e)
            task_info['completion_time'] = time.time()
        
        finally:
            # Clean up resources
            self.attention_manager.deallocate_resources(task_id)
            
            # Move to history
            self.task_history.append(task_info.copy())
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Update performance metrics
            self._update_performance_metrics(task_info)
    
    async def _run_task_callable(self, task_callable: Callable) -> Any:
        """Run a task callable, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(task_callable):
            return await task_callable()
        else:
            return task_callable()
    
    def _update_performance_metrics(self, task_info: Dict[str, Any]):
        """Update performance metrics based on completed task."""
        if 'completion_time' not in task_info or 'start_time' not in task_info:
            return
        
        execution_time = task_info['completion_time'] - task_info['start_time']
        
        # Update total tasks processed
        self.total_tasks_processed += 1
        
        # Update average completion time
        if self.total_tasks_processed == 1:
            self.average_completion_time = execution_time
        else:
            self.average_completion_time = (
                (self.average_completion_time * (self.total_tasks_processed - 1) + execution_time) 
                / self.total_tasks_processed
            )
        
        # Update resource efficiency (tasks completed per resource unit)
        resource_req = task_info['resource_req']
        resource_cost = resource_req.cpu_cycles + resource_req.memory_allocation
        if resource_cost > 0:
            task_efficiency = 1.0 / resource_cost
            self.resource_efficiency = (
                (self.resource_efficiency * (self.total_tasks_processed - 1) + task_efficiency)
                / self.total_tasks_processed
            )
    
    def get_kernel_statistics(self) -> Dict[str, Any]:
        """Get resource kernel performance statistics."""
        return {
            'active_tasks': len(self.active_tasks),
            'total_tasks_processed': self.total_tasks_processed,
            'average_completion_time': self.average_completion_time,
            'resource_efficiency': self.resource_efficiency,
            'task_history_size': len(self.task_history),
            'max_concurrent_tasks': self.max_concurrent_tasks
        }


# Integration utilities for existing tensor fragment system

def create_ecan_enhanced_tensor_fragment(tensor_arch, content: Any, 
                                       attention_value: AttentionValue,
                                       resource_req: ResourceAllocation) -> 'TensorFragment':
    """
    Create a tensor fragment enhanced with ECAN attention values.
    
    Args:
        tensor_arch: TensorFragmentArchitecture instance
        content: Content to encode
        attention_value: ECAN attention value
        resource_req: Resource requirements
    
    Returns:
        Enhanced TensorFragment with ECAN integration
    """
    from .tensor_fragments import TensorSignature, TensorFragment
    
    # Convert attention and resource values to tensor signature
    norm_sti, norm_lti, confidence = attention_value.normalize_for_tensor()
    modality, depth, salience, autonomy = resource_req.to_tensor_signature()
    
    # Create tensor signature using ECAN values
    # Map: [tasks, attention, priority, resources] → [modality, depth, context, salience, autonomy_index]
    signature = TensorSignature(
        modality=modality,      # Task type → modality
        depth=depth,            # Priority → depth  
        context=norm_lti,       # LTI → context
        salience=norm_sti,      # STI → salience
        autonomy_index=confidence  # Confidence → autonomy
    )
    
    # Create tensor fragment
    fragment = TensorFragment(signature, content)
    
    # Add ECAN metadata
    fragment.metadata['ecan_attention'] = attention_value
    fragment.metadata['resource_allocation'] = resource_req
    fragment.metadata['creation_timestamp'] = time.time()
    
    return fragment


def sync_tensor_fragments_with_ecan(tensor_arch, attention_manager: ECANAttentionManager):
    """
    Synchronize existing tensor fragments with ECAN attention values.
    
    Args:
        tensor_arch: TensorFragmentArchitecture instance
        attention_manager: ECAN attention manager
    """
    if not tensor_arch.atomspace:
        return
    
    # Iterate through all atoms in atomspace
    for atom_handle, atom_data in tensor_arch.atomspace.atoms.items():
        if 'tensor_fragment' in atom_data:
            # Extract tensor signature
            signature = atom_data['tensor_fragment']['signature']
            
            # Convert tensor values to attention values
            attention_value = AttentionValue.from_tensor_values(
                signature['salience'],      # STI from salience
                signature['context'],       # LTI from context  
                signature['autonomy_index'] # Confidence from autonomy
            )
            
            # Set in attention manager
            attention_manager.set_attention_value(atom_handle, attention_value)