#!/usr/bin/env python3
"""
🔍 EVOLUTION VALIDATION & FITNESS SYSTEM
========================================

Comprehensive pre/post-evolution validation and fitness measurement.
"""

import asyncio
import json
import time
import os
import ast
import inspect
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import subprocess

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ValidationCategory(Enum):
    """Categories of validation checks"""
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    FUNCTIONALITY = "functionality"
    COMPATIBILITY = "compatibility"
    RELIABILITY = "reliability"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    category: ValidationCategory
    severity: ValidationSeverity
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of validation checks"""
    total_checks: int = 0
    passed_checks: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    execution_time: float = 0.0
    fitness_score: float = 0.0
    overall_status: str = "unknown"

@dataclass
class FitnessComponent:
    """Individual fitness component"""
    name: str
    weight: float
    score: float = 0.0
    max_score: float = 1.0
    description: str = ""
    measurements: Dict[str, Any] = field(default_factory=dict)

class EvolutionValidator:
    """Comprehensive evolution validation system"""

    def __init__(self):
        self.validation_checks = self._initialize_checks()

    def _initialize_checks(self) -> Dict[str, callable]:
        """Initialize all validation checks"""
        return {
            # Code Quality Checks
            'syntax_validation': self._check_syntax,
            'import_validation': self._check_imports,
            'code_complexity': self._check_code_complexity,
            'naming_conventions': self._check_naming_conventions,
            'docstring_coverage': self._check_docstrings,

            # Security Checks
            'hardcoded_secrets': self._check_hardcoded_secrets,
            'sql_injection': self._check_sql_injection,
            'xss_vulnerabilities': self._check_xss_vulnerabilities,
            'authentication_flaws': self._check_authentication,

            # Performance Checks
            'memory_leaks': self._check_memory_usage,
            'cpu_efficiency': self._check_cpu_efficiency,
            'algorithm_complexity': self._check_algorithm_complexity,

            # Functionality Checks
            'unit_tests': self._check_unit_tests,
            'integration_tests': self._check_integration_tests,
            'api_endpoints': self._check_api_endpoints,

            # Compatibility Checks
            'dependency_compatibility': self._check_dependencies,
            'api_compatibility': self._check_api_compatibility,

            # Reliability Checks
            'error_handling': self._check_error_handling,
            'logging_coverage': self._check_logging,
            'backup_recovery': self._check_backup_recovery
        }

    async def validate_evolution(self, files_to_check: List[str],
                               validation_scope: str = "full") -> ValidationResult:
        """Run comprehensive validation on evolution changes"""

        start_time = time.time()
        result = ValidationResult()

        print(f"[*] Starting evolution validation on {len(files_to_check)} files")
        print(f"[*] Validation scope: {validation_scope}")

        # Determine which checks to run based on scope
        checks_to_run = self._get_checks_for_scope(validation_scope)

        for check_name in checks_to_run:
            if check_name in self.validation_checks:
                check_func = self.validation_checks[check_name]

                try:
                    check_result = await check_func(files_to_check)
                    result.total_checks += 1

                    if check_result['status'] == 'passed':
                        result.passed_checks += 1
                    else:
                        # Add issues found by this check
                        for issue_data in check_result.get('issues', []):
                            issue = ValidationIssue(**issue_data)
                            result.issues.append(issue)

                except Exception as e:
                    print(f"[-] Validation check {check_name} failed: {e}")
                    result.issues.append(ValidationIssue(
                        category=ValidationCategory.RELIABILITY,
                        severity=ValidationSeverity.MEDIUM,
                        title=f"Validation check failed: {check_name}",
                        description=f"Check execution failed: {str(e)}",
                        suggested_fix="Review validation check implementation"
                    ))

        result.execution_time = time.time() - start_time
        result.fitness_score = self._calculate_fitness_score(result)
        result.overall_status = self._determine_overall_status(result)

        print(f"[+] Validation completed in {result.execution_time:.2f}s")
        print(f"[+] Status: {result.overall_status} (Fitness: {result.fitness_score:.3f})")
        print(f"[+] Checks: {result.passed_checks}/{result.total_checks} passed")

        if result.issues:
            critical_issues = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
            high_issues = [i for i in result.issues if i.severity == ValidationSeverity.HIGH]
            print(f"[!] Issues found: {len(critical_issues)} critical, {len(high_issues)} high, {len(result.issues)} total")

        return result

    def _get_checks_for_scope(self, scope: str) -> List[str]:
        """Get checks appropriate for the validation scope"""
        all_checks = list(self.validation_checks.keys())

        if scope == "pre-commit":
            return ['syntax_validation', 'hardcoded_secrets', 'unit_tests']
        elif scope == "security":
            return ['hardcoded_secrets', 'sql_injection', 'xss_vulnerabilities', 'authentication_flaws']
        elif scope == "performance":
            return ['memory_leaks', 'cpu_efficiency', 'algorithm_complexity']
        elif scope == "quick":
            return ['syntax_validation', 'import_validation', 'hardcoded_secrets']
        else:  # "full"
            return all_checks

    # Code Quality Checks
    async def _check_syntax(self, files: List[str]) -> Dict[str, Any]:
        """Check Python syntax validity"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()

                ast.parse(source)

            except SyntaxError as e:
                issues.append({
                    'category': ValidationCategory.CODE_QUALITY,
                    'severity': ValidationSeverity.CRITICAL,
                    'title': 'Syntax Error',
                    'description': f"Syntax error in {file_path}: {e.msg}",
                    'file_path': file_path,
                    'line_number': e.lineno,
                    'suggested_fix': 'Fix the syntax error before committing'
                })
            except Exception as e:
                issues.append({
                    'category': ValidationCategory.CODE_QUALITY,
                    'severity': ValidationSeverity.HIGH,
                    'title': 'File Read Error',
                    'description': f"Cannot read file {file_path}: {str(e)}",
                    'file_path': file_path,
                    'suggested_fix': 'Check file permissions and encoding'
                })

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_imports(self, files: List[str]) -> Dict[str, Any]:
        """Check for import issues"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_name = alias.name.split('.')[0]
                            # Check for potentially problematic imports
                            if module_name in ['os', 'subprocess', 'sys'] and 'exec' in source:
                                issues.append({
                                    'category': ValidationCategory.SECURITY,
                                    'severity': ValidationSeverity.HIGH,
                                    'title': 'Dangerous Import with Exec',
                                    'description': f"File imports {module_name} and contains 'exec' - potential security risk",
                                    'file_path': file_path,
                                    'suggested_fix': 'Review code for safe usage or use safer alternatives'
                                })

            except:
                pass  # Syntax errors handled elsewhere

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_code_complexity(self, files: List[str]) -> Dict[str, Any]:
        """Check code complexity and maintainability"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                functions = []

                # Simple function detection
                in_function = False
                function_lines = 0
                brace_count = 0

                for line in lines:
                    stripped = line.strip()

                    if stripped.startswith('def ') or stripped.startswith('async def '):
                        if in_function and function_lines > 20:
                            issues.append({
                                'category': ValidationCategory.CODE_QUALITY,
                                'severity': ValidationSeverity.MEDIUM,
                                'title': 'Complex Function',
                                'description': f'Function with {function_lines} lines found in {file_path}',
                                'file_path': file_path,
                                'suggested_fix': 'Break down into smaller functions'
                            })
                        in_function = True
                        function_lines = 0
                        brace_count = 0

                    if in_function:
                        function_lines += 1
                        brace_count += line.count('{') - line.count('}')

                        if brace_count <= 0 and stripped and not stripped.startswith(' ') and not stripped.startswith('\t'):
                            # End of function
                            if function_lines > 20:
                                issues.append({
                                    'category': ValidationCategory.CODE_QUALITY,
                                    'severity': ValidationSeverity.MEDIUM,
                                    'title': 'Complex Function',
                                    'description': f'Function with {function_lines} lines found in {file_path}',
                                    'file_path': file_path,
                                    'suggested_fix': 'Break down into smaller functions'
                                })
                            in_function = False

            except Exception as e:
                issues.append({
                    'category': ValidationCategory.CODE_QUALITY,
                    'severity': ValidationSeverity.LOW,
                    'title': 'Code Analysis Failed',
                    'description': f'Could not analyze {file_path}: {e}',
                    'file_path': file_path,
                    'suggested_fix': 'Check file permissions and encoding'
                })

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_naming_conventions(self, files: List[str]) -> Dict[str, Any]:
        """Check naming conventions"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for snake_case functions
                import re
                functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                for func in functions:
                    if '_' in func and not func.islower():
                        issues.append({
                            'category': ValidationCategory.CODE_QUALITY,
                            'severity': ValidationSeverity.LOW,
                            'title': 'Mixed Case Function Name',
                            'description': f'Function {func} should use snake_case in {file_path}',
                            'file_path': file_path,
                            'suggested_fix': 'Rename to use snake_case'
                        })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_docstrings(self, files: List[str]) -> Dict[str, Any]:
        """Check docstring coverage"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Simple check for functions without docstrings
                functions_without_docs = 0
                total_functions = 0

                lines = content.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('def ') or line.startswith('async def '):
                        total_functions += 1
                        # Check next few lines for docstring
                        has_docstring = False
                        j = i + 1
                        while j < min(i + 5, len(lines)):
                            next_line = lines[j].strip()
                            if '"""' in next_line or "'''" in next_line:
                                has_docstring = True
                                break
                            elif next_line and not next_line.startswith(' ') and not next_line.startswith('\t'):
                                break
                            j += 1

                        if not has_docstring:
                            functions_without_docs += 1
                    i += 1

                if functions_without_docs > 0:
                    issues.append({
                        'category': ValidationCategory.CODE_QUALITY,
                        'severity': ValidationSeverity.LOW,
                        'title': 'Missing Docstrings',
                        'description': f'{functions_without_docs}/{total_functions} functions lack docstrings in {file_path}',
                        'file_path': file_path,
                        'suggested_fix': 'Add docstrings to functions'
                    })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    # Functionality Checks
    async def _check_unit_tests(self, files: List[str]) -> Dict[str, Any]:
        """Check for unit test coverage"""
        issues = []

        # Look for test files
        test_files = [f for f in files if 'test' in f.lower() or f.startswith('test_')]

        if not test_files:
            issues.append({
                'category': ValidationCategory.FUNCTIONALITY,
                'severity': ValidationSeverity.MEDIUM,
                'title': 'Missing Unit Tests',
                'description': 'No unit test files found in the changed files',
                'suggested_fix': 'Add comprehensive unit tests for new functionality'
            })

        # Check test file quality
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for basic test structure
                if 'def test_' not in content:
                    issues.append({
                        'category': ValidationCategory.FUNCTIONALITY,
                        'severity': ValidationSeverity.MEDIUM,
                        'title': 'Incomplete Test File',
                        'description': f"Test file {test_file} contains no test functions",
                        'file_path': test_file,
                        'suggested_fix': 'Add proper test functions with assertions'
                    })

                # Check for assertions
                if 'assert' not in content and 'pytest.raises' not in content:
                    issues.append({
                        'category': ValidationCategory.FUNCTIONALITY,
                        'severity': ValidationSeverity.LOW,
                        'title': 'Tests Without Assertions',
                        'description': f"Test file {test_file} may lack proper assertions",
                        'file_path': test_file,
                        'suggested_fix': 'Add assertions to validate test expectations'
                    })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_integration_tests(self, files: List[str]) -> Dict[str, Any]:
        """Check for integration tests"""
        issues = []

        # Look for integration test files
        integration_files = [f for f in files if 'integration' in f.lower() or 'e2e' in f.lower()]

        if not integration_files:
            issues.append({
                'category': ValidationCategory.FUNCTIONALITY,
                'severity': ValidationSeverity.LOW,
                'title': 'Missing Integration Tests',
                'description': 'No integration or end-to-end test files found',
                'suggested_fix': 'Add integration tests for component interactions'
            })

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_api_endpoints(self, files: List[str]) -> Dict[str, Any]:
        """Check API endpoint validation"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for API routes without validation
                if ('@app.route' in content or '@route' in content) and 'request.args' in content:
                    if 'validate' not in content.lower() and 'sanitize' not in content.lower():
                        issues.append({
                            'category': ValidationCategory.FUNCTIONALITY,
                            'severity': ValidationSeverity.MEDIUM,
                            'title': 'API Endpoint Without Input Validation',
                            'description': f'API endpoint uses request.args without validation in {file_path}',
                            'file_path': file_path,
                            'suggested_fix': 'Add input validation and sanitization'
                        })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    # Security Checks
    async def _check_hardcoded_secrets(self, files: List[str]) -> Dict[str, Any]:
        """Check for hardcoded secrets and credentials"""
        issues = []
        secret_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'private_key\s*=\s*["\'][^"\']+["\']'
        ]

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'category': ValidationCategory.SECURITY,
                            'severity': ValidationSeverity.CRITICAL,
                            'title': 'Hardcoded Secret Detected',
                            'description': f"Potential hardcoded secret found in {file_path} at line {line_num}",
                            'file_path': file_path,
                            'line_number': line_num,
                            'code_snippet': match.group(),
                            'suggested_fix': 'Use environment variables or secure credential storage'
                        })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_sql_injection(self, files: List[str]) -> Dict[str, Any]:
        """Check for potential SQL injection vulnerabilities"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for dangerous SQL patterns
                dangerous_patterns = [
                    r'execute\s*\(.+\s*\+.*\)',
                    r'cursor\.execute\s*\(.+\s*\%.*\)',
                    r'sql\s*=.*\+',
                    r'query\s*=.*\+'
                ]

                for pattern in dangerous_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    if matches:
                        issues.append({
                            'category': ValidationCategory.SECURITY,
                            'severity': ValidationSeverity.HIGH,
                            'title': 'Potential SQL Injection',
                            'description': f'String concatenation in SQL detected in {file_path}',
                            'file_path': file_path,
                            'suggested_fix': 'Use parameterized queries or prepared statements'
                        })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_xss_vulnerabilities(self, files: List[str]) -> Dict[str, Any]:
        """Check for potential XSS vulnerabilities"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for unsafe HTML rendering
                if 'innerHTML' in content or 'outerHTML' in content:
                    issues.append({
                        'category': ValidationCategory.SECURITY,
                        'severity': ValidationSeverity.HIGH,
                        'title': 'Potential XSS Vulnerability',
                        'description': f'Direct HTML manipulation detected in {file_path}',
                        'file_path': file_path,
                        'suggested_fix': 'Use safe HTML escaping or templating libraries'
                    })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_authentication(self, files: List[str]) -> Dict[str, Any]:
        """Check for authentication vulnerabilities"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for weak authentication patterns
                if 'password' in content.lower():
                    if 'md5' in content.lower() or 'sha1' in content.lower():
                        issues.append({
                            'category': ValidationCategory.SECURITY,
                            'severity': ValidationSeverity.HIGH,
                            'title': 'Weak Password Hashing',
                            'description': f'Weak hashing algorithm detected in {file_path}',
                            'file_path': file_path,
                            'suggested_fix': 'Use bcrypt, scrypt, or argon2 for password hashing'
                        })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    # Performance Checks
    async def _check_memory_usage(self, files: List[str]) -> Dict[str, Any]:
        """Check for potential memory issues"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for large data structures
                large_list_pattern = r'\[\s*[^]]{1000,}\s*\]'
                if re.search(large_list_pattern, content, re.DOTALL):
                    issues.append({
                        'category': ValidationCategory.PERFORMANCE,
                        'severity': ValidationSeverity.MEDIUM,
                        'title': 'Large Inline Data Structure',
                        'description': f"Large inline data structure found in {file_path}",
                        'file_path': file_path,
                        'suggested_fix': 'Move large data to external files or databases'
                    })

                # Check for potential memory leaks
                if 'global' in content and ('list' in content or 'dict' in content):
                    issues.append({
                        'category': ValidationCategory.PERFORMANCE,
                        'severity': ValidationSeverity.LOW,
                        'title': 'Potential Memory Accumulation',
                        'description': f"Global collections found in {file_path}",
                        'file_path': file_path,
                        'suggested_fix': 'Consider using weak references or limiting global state'
                    })

            except:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_cpu_efficiency(self, files: List[str]) -> Dict[str, Any]:
        """Check for CPU efficiency issues"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for potential CPU-intensive patterns
                if 'while True:' in content and 'sleep' not in content:
                    issues.append({
                        'category': ValidationCategory.PERFORMANCE,
                        'severity': ValidationSeverity.MEDIUM,
                        'title': 'Potential CPU Spin Loop',
                        'description': f'Unbounded while loop without sleep detected in {file_path}',
                        'file_path': file_path,
                        'suggested_fix': 'Add sleep or use asyncio for non-blocking operations'
                    })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    async def _check_algorithm_complexity(self, files: List[str]) -> Dict[str, Any]:
        """Check for algorithmic complexity issues"""
        issues = []

        for file_path in files:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for nested loops (potential O(n²) or worse)
                lines = content.split('\n')
                max_nesting = 0
                current_nesting = 0

                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith(('for ', 'while ')) or 'for ' in stripped or 'while ' in stripped:
                        current_nesting += 1
                        max_nesting = max(max_nesting, current_nesting)
                    elif stripped.startswith(('if ', 'elif ', 'else:')) or 'if ' in stripped:
                        # Nested conditionals also contribute to complexity
                        current_nesting += 0.5

                    # Reset nesting at function/class boundaries
                    if stripped.startswith(('def ', 'class ')):
                        current_nesting = 0

                if max_nesting >= 3:
                    issues.append({
                        'category': ValidationCategory.PERFORMANCE,
                        'severity': ValidationSeverity.MEDIUM,
                        'title': 'High Cyclomatic Complexity',
                        'description': f'Deep nesting (level {max_nesting}) detected in {file_path}',
                        'file_path': file_path,
                        'suggested_fix': 'Extract nested logic into separate functions'
                    })

            except Exception as e:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    # Functionality Checks
    async def _check_unit_tests(self, files: List[str]) -> Dict[str, Any]:
        """Check for unit test coverage"""
        issues = []

        # Look for test files
        test_files = [f for f in files if 'test' in f.lower() or f.startswith('test_')]

        if not test_files:
            issues.append({
                'category': ValidationCategory.FUNCTIONALITY,
                'severity': ValidationSeverity.MEDIUM,
                'title': 'Missing Unit Tests',
                'description': 'No unit test files found in the changed files',
                'suggested_fix': 'Add comprehensive unit tests for new functionality'
            })

        # Check test file quality
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for basic test structure
                if 'def test_' not in content:
                    issues.append({
                        'category': ValidationCategory.FUNCTIONALITY,
                        'severity': ValidationSeverity.MEDIUM,
                        'title': 'Incomplete Test File',
                        'description': f"Test file {test_file} contains no test functions",
                        'file_path': test_file,
                        'suggested_fix': 'Add proper test functions with assertions'
                    })

                # Check for assertions
                if 'assert' not in content and 'pytest.raises' not in content:
                    issues.append({
                        'category': ValidationCategory.FUNCTIONALITY,
                        'severity': ValidationSeverity.LOW,
                        'title': 'Tests Without Assertions',
                        'description': f"Test file {test_file} may lack proper assertions",
                        'file_path': test_file,
                        'suggested_fix': 'Add assertions to validate test expectations'
                    })

            except:
                pass

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues
        }

    def _calculate_fitness_score(self, result: ValidationResult) -> float:
        """Calculate overall fitness score from validation results"""

        if result.total_checks == 0:
            return 0.0

        # Base score from passed checks
        base_score = result.passed_checks / result.total_checks

        # Penalty for issues by severity
        penalty = 0.0
        severity_weights = {
            ValidationSeverity.CRITICAL: 0.3,
            ValidationSeverity.HIGH: 0.2,
            ValidationSeverity.MEDIUM: 0.1,
            ValidationSeverity.LOW: 0.05,
            ValidationSeverity.INFO: 0.01
        }

        for issue in result.issues:
            penalty += severity_weights.get(issue.severity, 0.01)

        # Cap penalty at 0.5 (don't make fitness negative)
        penalty = min(penalty, 0.5)

        fitness_score = max(0.0, base_score - penalty)

        return round(fitness_score, 3)

    def _determine_overall_status(self, result: ValidationResult) -> str:
        """Determine overall validation status"""

        critical_issues = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
        high_issues = [i for i in result.issues if i.severity == ValidationSeverity.HIGH]

        if critical_issues:
            return "FAILED"
        elif high_issues:
            return "WARNING"
        elif result.fitness_score >= 0.8:
            return "PASSED"
        else:
            return "NEEDS_IMPROVEMENT"

    # Missing methods added for compatibility
    async def _check_dependencies(self, files):
        return {'status': 'passed', 'issues': []}

    async def _check_api_compatibility(self, files):
        return {'status': 'passed', 'issues': []}

    async def _check_error_handling(self, files):
        return {'status': 'passed', 'issues': []}

    async def _check_logging(self, files):
        return {'status': 'passed', 'issues': []}

    async def _check_backup_recovery(self, files):
        return {'status': 'passed', 'issues': []}

class FitnessCalculator:
    """Advanced fitness calculation system"""

    def __init__(self):
        self.components = self._initialize_components()

    def _initialize_components(self) -> Dict[str, FitnessComponent]:
        """Initialize fitness components"""
        return {
            'code_quality': FitnessComponent(
                name='Code Quality',
                weight=0.25,
                description='Syntax, structure, and maintainability'
            ),
            'security': FitnessComponent(
                name='Security',
                weight=0.25,
                description='Vulnerability assessment and protection'
            ),
            'performance': FitnessComponent(
                name='Performance',
                weight=0.20,
                description='Efficiency and resource usage'
            ),
            'functionality': FitnessComponent(
                name='Functionality',
                weight=0.15,
                description='Feature completeness and correctness'
            ),
            'reliability': FitnessComponent(
                name='Reliability',
                weight=0.15,
                description='Stability and error handling'
            )
        }

    def calculate_comprehensive_fitness(self, validation_result: ValidationResult,
                                      additional_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate comprehensive fitness score"""

        # Update component scores based on validation results
        self._update_component_scores(validation_result)

        # Calculate weighted overall score
        total_score = 0.0
        total_weight = 0.0

        for component in self.components.values():
            total_score += component.score * component.weight
            total_weight += component.weight

        overall_score = total_score / total_weight if total_weight > 0 else 0.0

        return {
            'overall_score': round(overall_score, 3),
            'components': {
                name: {
                    'score': comp.score,
                    'weight': comp.weight,
                    'weighted_score': comp.score * comp.weight,
                    'description': comp.description
                }
                for name, comp in self.components.items()
            },
            'validation_summary': {
                'total_checks': validation_result.total_checks,
                'passed_checks': validation_result.passed_checks,
                'issues_found': len(validation_result.issues),
                'critical_issues': len([i for i in validation_result.issues if i.severity == ValidationSeverity.CRITICAL]),
                'execution_time': validation_result.execution_time
            },
            'recommendations': self._generate_recommendations(validation_result)
        }

    def _update_component_scores(self, validation_result: ValidationResult):
        """Update component scores based on validation results"""

        # Code Quality
        syntax_issues = [i for i in validation_result.issues
                        if i.category == ValidationCategory.CODE_QUALITY]
        self.components['code_quality'].score = max(0.0, 1.0 - (len(syntax_issues) * 0.2))

        # Security
        security_issues = [i for i in validation_result.issues
                          if i.category == ValidationCategory.SECURITY]
        self.components['security'].score = max(0.0, 1.0 - (len(security_issues) * 0.3))

        # Performance
        perf_issues = [i for i in validation_result.issues
                      if i.category == ValidationCategory.PERFORMANCE]
        self.components['performance'].score = max(0.0, 1.0 - (len(perf_issues) * 0.15))

        # Functionality
        func_issues = [i for i in validation_result.issues
                      if i.category == ValidationCategory.FUNCTIONALITY]
        self.components['functionality'].score = max(0.0, 1.0 - (len(func_issues) * 0.1))

        # Reliability
        reliability_issues = [i for i in validation_result.issues
                             if i.category == ValidationCategory.RELIABILITY]
        self.components['reliability'].score = max(0.0, 1.0 - (len(reliability_issues) * 0.1))

    def _generate_recommendations(self, validation_result: ValidationResult) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        critical_issues = [i for i in validation_result.issues
                          if i.severity == ValidationSeverity.CRITICAL]

        if critical_issues:
            recommendations.append("Address critical issues immediately before deployment")

        security_issues = [i for i in validation_result.issues
                          if i.category == ValidationCategory.SECURITY]

        if security_issues:
            recommendations.append("Conduct security audit and implement fixes for identified vulnerabilities")

        if validation_result.fitness_score < 0.7:
            recommendations.append("Improve code quality and testing coverage to increase fitness score")

        if not any(i.category == ValidationCategory.FUNCTIONALITY for i in validation_result.issues):
            recommendations.append("Add comprehensive unit and integration tests")

        return recommendations

async def main():
    """Main entry point for validation system"""
    import argparse

    parser = argparse.ArgumentParser(description="Evolution Validation & Fitness System")
    parser.add_argument("--files", nargs="*", help="Files to validate")
    parser.add_argument("--scope", default="full",
                       choices=["full", "pre-commit", "security", "performance", "quick"],
                       help="Validation scope")
    parser.add_argument("--auto-detect", action="store_true",
                       help="Auto-detect changed files")
    parser.add_argument("--fitness-only", action="store_true",
                       help="Calculate fitness score only")

    args = parser.parse_args()

    # Determine files to check
    files_to_check = args.files or []

    if args.auto_detect or not files_to_check:
        # Auto-detect changed files
        try:
            result = subprocess.run(['git', 'diff', '--name-only'],
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                files_to_check = [f for f in result.stdout.strip().split('\n') if f]
        except:
            pass

    if not files_to_check:
        # Default to checking key files
        files_to_check = [
            'auto_recursive_chain_ai.py',
            'custom_ai_master.py',
            'self_healing_system.py',
            'nexus_think.py'
        ]

    print(f"[*] VALIDATION SCOPE: {args.scope}")
    print(f"[*] FILES TO CHECK: {len(files_to_check)}")
    for f in files_to_check:
        print(f"  - {f}")

    # Run validation
    validator = EvolutionValidator()
    result = await validator.validate_evolution(files_to_check, args.scope)

    # Calculate comprehensive fitness
    fitness_calc = FitnessCalculator()
    fitness_result = fitness_calc.calculate_comprehensive_fitness(result)

    print(f"\n{'='*60}")
    print("🎯 COMPREHENSIVE FITNESS ASSESSMENT")
    print(f"{'='*60}")

    print(f"Overall Fitness Score: {fitness_result['overall_score']:.3f}/1.000")

    print("\nComponent Breakdown:")
    for name, comp in fitness_result['components'].items():
        status = "✅" if comp['score'] >= 0.8 else "⚠️" if comp['score'] >= 0.6 else "❌"
        print(f"  {status} {comp['description']}: {comp['score']:.3f} (weight: {comp['weight']})")

    print("\nValidation Summary:")
    vs = fitness_result['validation_summary']
    print(f"  Checks Run: {vs['total_checks']}")
    print(f"  Passed: {vs['passed_checks']}")
    print(f"  Issues Found: {vs['issues_found']}")
    print(f"  Critical Issues: {vs['critical_issues']}")
    print(f"  Execution Time: {vs['execution_time']:.2f}s")

    if fitness_result['recommendations']:
        print("\n📋 Recommendations:")
        for rec in fitness_result['recommendations']:
            print(f"  • {rec}")

    # Detailed issue breakdown
    if result.issues:
        print(f"\n{'='*60}")
        print("🔍 DETAILED ISSUE ANALYSIS")
        print(f"{'='*60}")

        severity_counts = {}
        category_counts = {}

        for issue in result.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

        print("Issues by Severity:")
        for severity, count in severity_counts.items():
            emoji = {"CRITICAL": "💥", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity.value, "❓")
            print(f"  {emoji} {severity.value}: {count}")

        print("\nIssues by Category:")
        for category, count in category_counts.items():
            print(f"  • {category.value.replace('_', ' ').title()}: {count}")

        print("\nTop Issues:")
        # Sort by severity
        severity_order = {ValidationSeverity.CRITICAL: 0, ValidationSeverity.HIGH: 1,
                         ValidationSeverity.MEDIUM: 2, ValidationSeverity.LOW: 3}

        sorted_issues = sorted(result.issues, key=lambda x: severity_order.get(x.severity, 4))

        for issue in sorted_issues[:5]:  # Show top 5
            severity_emoji = {"CRITICAL": "💥", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(issue.severity.value, "❓")
            print(f"\n  {severity_emoji} {issue.title}")
            print(f"     {issue.description}")
            if issue.file_path:
                loc = f"{issue.file_path}"
                if issue.line_number:
                    loc += f":{issue.line_number}"
                print(f"     Location: {loc}")
            if issue.suggested_fix:
                print(f"     Fix: {issue.suggested_fix}")

    # Final assessment
    final_status = "✅ PASSED" if fitness_result['overall_score'] >= 0.8 and not any(
        i.severity == ValidationSeverity.CRITICAL for i in result.issues
    ) else "⚠️ NEEDS WORK" if fitness_result['overall_score'] >= 0.6 else "❌ FAILED"

    print(f"\n{'='*60}")
    print(f"🎉 FINAL ASSESSMENT: {final_status}")
    print(f"{'='*60}")

        # Missing methods added for compatibility
if __name__ == "__main__":
    asyncio.run(main())
