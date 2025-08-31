"""
Scheme Cognitive Grammar Microservices for OpenCog integration.

This module implements Scheme-like cognitive grammar operations as microservices
that can parse, evaluate, and transform cognitive expressions in a functional style.
"""

import re
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio


class ExpressionType(Enum):
    """Types of cognitive grammar expressions."""
    ATOM = "atom"
    LIST = "list"
    LAMBDA = "lambda"
    DEFINITION = "definition"
    EVALUATION = "evaluation"
    PATTERN = "pattern"


@dataclass
class SchemeExpression:
    """
    Represents a parsed Scheme-like cognitive expression.
    """
    type: ExpressionType
    value: Any
    children: List['SchemeExpression']
    metadata: Dict[str, Any]
    
    def __str__(self):
        if self.type == ExpressionType.ATOM:
            return str(self.value)
        elif self.type == ExpressionType.LIST:
            child_strs = [str(child) for child in self.children]
            return f"({' '.join(child_strs)})"
        else:
            return f"({self.type.value} {self.value})"


class SchemeParser:
    """
    Parser for Scheme-like cognitive grammar expressions.
    """
    
    def __init__(self):
        self.tokens = []
        self.position = 0
    
    def tokenize(self, expression: str) -> List[str]:
        """
        Tokenize a Scheme expression string.
        
        Args:
            expression: The expression string to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenizer that handles parentheses, quotes, and whitespace
        token_pattern = r'\(|\)|"[^"]*"|[^\s()]+'
        tokens = re.findall(token_pattern, expression)
        return tokens
    
    def parse(self, expression: str) -> SchemeExpression:
        """
        Parse a Scheme expression string into a SchemeExpression tree.
        
        Args:
            expression: The expression string to parse
            
        Returns:
            The parsed SchemeExpression
        """
        self.tokens = self.tokenize(expression.strip())
        self.position = 0
        
        if not self.tokens:
            return SchemeExpression(ExpressionType.ATOM, None, [], {})
        
        return self._parse_expression()
    
    def _parse_expression(self) -> SchemeExpression:
        """Parse a single expression."""
        if self.position >= len(self.tokens):
            return SchemeExpression(ExpressionType.ATOM, None, [], {})
        
        token = self.tokens[self.position]
        
        if token == '(':
            return self._parse_list()
        else:
            return self._parse_atom(token)
    
    def _parse_list(self) -> SchemeExpression:
        """Parse a list expression."""
        self.position += 1  # Skip '('
        
        children = []
        while self.position < len(self.tokens) and self.tokens[self.position] != ')':
            child = self._parse_expression()
            children.append(child)
        
        if self.position < len(self.tokens):
            self.position += 1  # Skip ')'
        
        # Determine expression type based on first element
        if children:
            first_child = children[0]
            if isinstance(first_child.value, str):
                if first_child.value in ['lambda', 'define', 'let']:
                    expr_type = ExpressionType.DEFINITION
                elif first_child.value in ['eval', 'apply']:
                    expr_type = ExpressionType.EVALUATION
                elif first_child.value in ['match', 'pattern']:
                    expr_type = ExpressionType.PATTERN
                else:
                    expr_type = ExpressionType.LIST
            else:
                expr_type = ExpressionType.LIST
        else:
            expr_type = ExpressionType.LIST
        
        return SchemeExpression(expr_type, None, children, {})
    
    def _parse_atom(self, token: str) -> SchemeExpression:
        """Parse an atomic expression."""
        self.position += 1
        
        # Handle quoted strings
        if token.startswith('"') and token.endswith('"'):
            value = token[1:-1]  # Remove quotes
        # Try to parse as number
        elif token.replace('.', '').replace('-', '').isdigit():
            value = float(token) if '.' in token else int(token)
        # Boolean values
        elif token.lower() in ['#t', 'true']:
            value = True
        elif token.lower() in ['#f', 'false']:
            value = False
        # Symbol
        else:
            value = token
        
        return SchemeExpression(ExpressionType.ATOM, value, [], {})


class CognitiveGrammarMicroservice:
    """
    Microservice for cognitive grammar operations.
    """
    
    def __init__(self, atomspace=None):
        """
        Initialize the cognitive grammar microservice.
        
        Args:
            atomspace: Optional AtomSpace instance for hypergraph operations
        """
        self.atomspace = atomspace
        self.parser = SchemeParser()
        self.environment = {}
        self.built_in_functions = self._create_built_in_functions()
    
    def _create_built_in_functions(self) -> Dict[str, Callable]:
        """Create built-in cognitive grammar functions."""
        
        def cog_define(name, value):
            """Define a cognitive concept or function."""
            self.environment[name] = value
            if self.atomspace:
                # Create concept node in AtomSpace
                concept_node = self.atomspace.add_node("ConceptNode", name)
                value_node = self.atomspace.add_node("ConceptNode", str(value))
                self.atomspace.add_link("DefineLink", [concept_node, value_node])
            return value
        
        def cog_eval(expression):
            """Evaluate a cognitive expression."""
            if isinstance(expression, SchemeExpression):
                return self.evaluate_expression(expression)
            return expression
        
        def cog_match(pattern, target):
            """Match a cognitive pattern against a target."""
            # Simple pattern matching - can be enhanced
            if str(pattern) == str(target):
                return True
            return False
        
        def cog_apply(function, arguments):
            """Apply a function to arguments."""
            if callable(function):
                return function(*arguments)
            return None
        
        def cog_lambda(params, body):
            """Create a lambda function."""
            def lambda_func(*args):
                # Create local environment
                local_env = dict(zip(params, args))
                old_env = self.environment.copy()
                self.environment.update(local_env)
                try:
                    result = self.evaluate_expression(body)
                finally:
                    self.environment = old_env
                return result
            return lambda_func
        
        def cog_let(bindings, body):
            """Let expression with local bindings."""
            old_env = self.environment.copy()
            try:
                # Apply bindings
                for binding in bindings:
                    if len(binding.children) == 2:
                        var_name = binding.children[0].value
                        var_value = self.evaluate_expression(binding.children[1])
                        self.environment[var_name] = var_value
                
                # Evaluate body
                return self.evaluate_expression(body)
            finally:
                self.environment = old_env
        
        def cog_if(condition, then_expr, else_expr=None):
            """Conditional expression."""
            if self.evaluate_expression(condition):
                return self.evaluate_expression(then_expr)
            elif else_expr:
                return self.evaluate_expression(else_expr)
            return None
        
        def cog_and(*expressions):
            """Logical AND operation."""
            for expr in expressions:
                if not self.evaluate_expression(expr):
                    return False
            return True
        
        def cog_or(*expressions):
            """Logical OR operation."""
            for expr in expressions:
                if self.evaluate_expression(expr):
                    return True
            return False
        
        def cog_not(expression):
            """Logical NOT operation."""
            return not self.evaluate_expression(expression)
        
        return {
            'define': cog_define,
            'eval': cog_eval,
            'match': cog_match,
            'apply': cog_apply,
            'lambda': cog_lambda,
            'let': cog_let,
            'if': cog_if,
            'and': cog_and,
            'or': cog_or,
            'not': cog_not,
            '+': lambda *args: sum(args),
            '-': lambda a, b=None: -a if b is None else a - b,
            '*': lambda *args: eval('*'.join(map(str, args))),
            '/': lambda a, b: a / b,
            '=': lambda a, b: a == b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            'cons': lambda a, b: [a] + (b if isinstance(b, list) else [b]),
            'car': lambda lst: lst[0] if lst else None,
            'cdr': lambda lst: lst[1:] if len(lst) > 1 else [],
            'null?': lambda x: x is None or (isinstance(x, list) and len(x) == 0)
        }
    
    def evaluate_expression(self, expression: SchemeExpression) -> Any:
        """
        Evaluate a cognitive grammar expression.
        
        Args:
            expression: The SchemeExpression to evaluate
            
        Returns:
            The evaluation result
        """
        if expression.type == ExpressionType.ATOM:
            value = expression.value
            
            # Look up in environment first
            if isinstance(value, str) and value in self.environment:
                return self.environment[value]
            
            # Return literal values
            return value
        
        elif expression.type in [ExpressionType.LIST, ExpressionType.EVALUATION, 
                                ExpressionType.DEFINITION, ExpressionType.PATTERN]:
            if not expression.children:
                return []
            
            # Get function/operator
            func_expr = expression.children[0]
            func_name = func_expr.value if func_expr.type == ExpressionType.ATOM else None
            
            # Special forms that don't evaluate all arguments
            if func_name in ['define', 'lambda', 'let', 'if']:
                return self._evaluate_special_form(func_name, expression.children[1:])
            
            # Regular function application
            if func_name in self.built_in_functions:
                func = self.built_in_functions[func_name]
                args = [self.evaluate_expression(child) for child in expression.children[1:]]
                return func(*args)
            
            # User-defined function
            elif func_name in self.environment:
                func = self.environment[func_name]
                if callable(func):
                    args = [self.evaluate_expression(child) for child in expression.children[1:]]
                    return func(*args)
            
            # Return as list if no function found
            return [self.evaluate_expression(child) for child in expression.children]
        
        return None
    
    def _evaluate_special_form(self, form_name: str, arguments: List[SchemeExpression]) -> Any:
        """Evaluate special forms that have non-standard evaluation rules."""
        
        if form_name == 'define':
            if len(arguments) >= 2:
                name = arguments[0].value
                value = self.evaluate_expression(arguments[1])
                return self.built_in_functions['define'](name, value)
        
        elif form_name == 'lambda':
            if len(arguments) >= 2:
                params = [child.value for child in arguments[0].children]
                body = arguments[1]
                return self.built_in_functions['lambda'](params, body)
        
        elif form_name == 'let':
            if len(arguments) >= 2:
                bindings = arguments[0].children
                body = arguments[1]
                return self.built_in_functions['let'](bindings, body)
        
        elif form_name == 'if':
            if len(arguments) >= 2:
                condition = arguments[0]
                then_expr = arguments[1]
                else_expr = arguments[2] if len(arguments) > 2 else None
                return self.built_in_functions['if'](condition, then_expr, else_expr)
        
        return None
    
    async def parse_and_evaluate(self, expression_str: str) -> Any:
        """
        Parse and evaluate a cognitive grammar expression string.
        
        Args:
            expression_str: The expression string to parse and evaluate
            
        Returns:
            The evaluation result
        """
        try:
            parsed = self.parser.parse(expression_str)
            result = self.evaluate_expression(parsed)
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_cognitive_binding(self, name: str, tensor_fragment) -> str:
        """
        Create a cognitive binding between a name and tensor fragment.
        
        Args:
            name: The binding name
            tensor_fragment: The TensorFragment to bind
            
        Returns:
            AtomSpace handle for the binding or None if no AtomSpace
        """
        if not self.atomspace:
            self.environment[name] = tensor_fragment
            return name
        
        # Encode the tensor fragment to hypergraph
        fragment_handle = tensor_fragment.encode_to_hypergraph(self.atomspace)
        
        # Create binding in AtomSpace
        name_node = self.atomspace.add_node("BindingNode", name)
        binding_link = self.atomspace.add_link("CognitiveBindingLink", [name_node, fragment_handle])
        
        # Also store in local environment
        self.environment[name] = tensor_fragment
        
        return binding_link
    
    def lookup_cognitive_binding(self, name: str):
        """
        Look up a cognitive binding by name.
        
        Args:
            name: The binding name to look up
            
        Returns:
            The bound value or None if not found
        """
        return self.environment.get(name)


class CognitiveGrammarServer:
    """
    Microservice server for cognitive grammar operations.
    """
    
    def __init__(self, atomspace=None, port: int = 8080):
        """
        Initialize the cognitive grammar server.
        
        Args:
            atomspace: Optional AtomSpace instance
            port: Port number for the microservice
        """
        self.atomspace = atomspace
        self.port = port
        self.microservice = CognitiveGrammarMicroservice(atomspace)
        self.running = False
        self.request_queue = asyncio.Queue()
        self.response_callbacks = {}
    
    async def start(self):
        """Start the cognitive grammar microservice server."""
        self.running = True
        print(f"🧠 Cognitive Grammar Microservice starting on port {self.port}")
        
        # Start the request processing loop
        asyncio.create_task(self._process_requests())
        
        print("✅ Cognitive Grammar Microservice ready")
    
    async def stop(self):
        """Stop the cognitive grammar microservice server."""
        self.running = False
        print("🛑 Cognitive Grammar Microservice stopped")
    
    async def _process_requests(self):
        """Process incoming requests."""
        while self.running:
            try:
                request_id, operation, data = await asyncio.wait_for(
                    self.request_queue.get(), timeout=1.0
                )
                
                # Process the request
                result = await self._handle_request(operation, data)
                
                # Send response if callback exists
                if request_id in self.response_callbacks:
                    callback = self.response_callbacks[request_id]
                    await callback(result)
                    del self.response_callbacks[request_id]
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing request: {e}")
    
    async def _handle_request(self, operation: str, data: Any) -> Any:
        """
        Handle a specific cognitive grammar request.
        
        Args:
            operation: The operation to perform
            data: The data for the operation
            
        Returns:
            The operation result
        """
        if operation == "parse":
            return self.microservice.parser.parse(data)
        
        elif operation == "evaluate":
            return await self.microservice.parse_and_evaluate(data)
        
        elif operation == "define_cognitive_concept":
            name = data.get("name")
            tensor_fragment = data.get("tensor_fragment")
            if name and tensor_fragment:
                return self.microservice.create_cognitive_binding(name, tensor_fragment)
            return None
        
        elif operation == "query_cognitive_concept":
            name = data.get("name")
            return self.microservice.lookup_cognitive_binding(name)
        
        elif operation == "transform_to_atomspace":
            expression_str = data.get("expression")
            if expression_str:
                parsed = self.microservice.parser.parse(expression_str)
                return await self._transform_to_atomspace(parsed)
            return None
        
        else:
            return f"Unknown operation: {operation}"
    
    async def _transform_to_atomspace(self, expression: SchemeExpression) -> Optional[str]:
        """
        Transform a Scheme expression to AtomSpace representation.
        
        Args:
            expression: The SchemeExpression to transform
            
        Returns:
            AtomSpace handle for the created structure
        """
        if not self.atomspace:
            return None
        
        if expression.type == ExpressionType.ATOM:
            # Create a ConceptNode for atomic values
            return self.atomspace.add_node("ConceptNode", str(expression.value))
        
        elif expression.type == ExpressionType.LIST:
            if not expression.children:
                return self.atomspace.add_link("ListLink", [])
            
            # Transform children recursively
            child_handles = []
            for child in expression.children:
                child_handle = await self._transform_to_atomspace(child)
                if child_handle:
                    child_handles.append(child_handle)
            
            # Determine link type based on first element
            if expression.children and expression.children[0].type == ExpressionType.ATOM:
                first_value = str(expression.children[0].value).lower()
                
                if first_value in ['evaluation', 'eval']:
                    return self.atomspace.add_link("EvaluationLink", child_handles[1:])
                elif first_value in ['inheritance', 'inherit']:
                    return self.atomspace.add_link("InheritanceLink", child_handles[1:])
                elif first_value in ['member', 'membership']:
                    return self.atomspace.add_link("MemberLink", child_handles[1:])
                elif first_value in ['similarity', 'similar']:
                    return self.atomspace.add_link("SimilarityLink", child_handles[1:])
            
            # Default to ListLink
            return self.atomspace.add_link("ListLink", child_handles)
        
        elif expression.type == ExpressionType.DEFINITION:
            # Handle definitions specially
            if expression.children and len(expression.children) >= 3:
                def_type = expression.children[0].value
                if def_type == "define":
                    name = str(expression.children[1].value)
                    value_handle = await self._transform_to_atomspace(expression.children[2])
                    if value_handle:
                        name_node = self.atomspace.add_node("ConceptNode", name)
                        return self.atomspace.add_link("DefineLink", [name_node, value_handle])
        
        return None
    
    async def submit_request(self, operation: str, data: Any) -> Any:
        """
        Submit a request to the cognitive grammar microservice.
        
        Args:
            operation: The operation to perform
            data: The data for the operation
            
        Returns:
            The operation result
        """
        request_id = f"req_{asyncio.current_task().get_name()}_{id(data)}"
        
        # Create response future
        response_future = asyncio.Future()
        self.response_callbacks[request_id] = lambda result: response_future.set_result(result)
        
        # Submit request
        await self.request_queue.put((request_id, operation, data))
        
        # Wait for response
        try:
            result = await asyncio.wait_for(response_future, timeout=10.0)
            return result
        except asyncio.TimeoutError:
            if request_id in self.response_callbacks:
                del self.response_callbacks[request_id]
            return "Request timeout"


class CognitiveGrammarOrchestrator:
    """
    Orchestrates multiple cognitive grammar microservices for complex operations.
    """
    
    def __init__(self, atomspace=None):
        """
        Initialize the cognitive grammar orchestrator.
        
        Args:
            atomspace: Optional AtomSpace instance
        """
        self.atomspace = atomspace
        self.services = {}
        self.service_discovery = {}
    
    def register_service(self, name: str, service: CognitiveGrammarServer):
        """
        Register a cognitive grammar microservice.
        
        Args:
            name: The service name
            service: The CognitiveGrammarServer instance
        """
        self.services[name] = service
        self.service_discovery[name] = {
            'port': service.port,
            'capabilities': ['parse', 'evaluate', 'transform_to_atomspace']
        }
    
    async def route_request(self, service_name: str, operation: str, data: Any) -> Any:
        """
        Route a request to a specific cognitive grammar service.
        
        Args:
            service_name: The name of the service to route to
            operation: The operation to perform
            data: The data for the operation
            
        Returns:
            The operation result
        """
        if service_name not in self.services:
            return f"Service not found: {service_name}"
        
        service = self.services[service_name]
        return await service.submit_request(operation, data)
    
    async def parallel_evaluate(self, expressions: List[str]) -> List[Any]:
        """
        Evaluate multiple expressions in parallel across services.
        
        Args:
            expressions: List of expression strings to evaluate
            
        Returns:
            List of evaluation results
        """
        tasks = []
        
        # Distribute expressions across available services
        service_names = list(self.services.keys())
        if not service_names:
            return []
        
        for i, expression in enumerate(expressions):
            service_name = service_names[i % len(service_names)]
            task = self.route_request(service_name, "evaluate", expression)
            tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def create_cognitive_grammar_expression(self, operation: str, *args) -> str:
        """
        Create a cognitive grammar expression string.
        
        Args:
            operation: The operation name
            *args: The operation arguments
            
        Returns:
            The formatted expression string
        """
        arg_strs = []
        for arg in args:
            if isinstance(arg, str):
                arg_strs.append(f'"{arg}"')
            else:
                arg_strs.append(str(arg))
        
        return f"({operation} {' '.join(arg_strs)})"
    
    def get_service_status(self) -> Dict[str, Dict]:
        """
        Get status of all registered services.
        
        Returns:
            Dictionary with service status information
        """
        status = {}
        for name, service in self.services.items():
            status[name] = {
                'running': service.running,
                'port': service.port,
                'queue_size': service.request_queue.qsize(),
                'pending_callbacks': len(service.response_callbacks)
            }
        return status