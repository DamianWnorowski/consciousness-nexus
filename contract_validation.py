#!/usr/bin/env python3
"""
🛡️ CONTRACT VALIDATION SYSTEM
==============================

JSON schema validation for evolution contracts to prevent injection attacks.
"""

import json
import jsonschema
import hashlib
import hmac
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import ast
import os

class ValidationSeverity(Enum):
    """Validation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of contract validation"""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    security_score: float = 0.0
    schema_version: str = "1.0"

@dataclass
class SecurityAnalysis:
    """Security analysis of contract"""
    injection_risks: List[str] = field(default_factory=list)
    unsafe_patterns: List[str] = field(default_factory=list)
    privilege_escalation_risks: List[str] = field(default_factory=list)
    data_exposure_risks: List[str] = field(default_factory=list)
    overall_risk_score: float = 0.0

class EvolutionContractValidator:
    """Comprehensive validator for evolution contracts"""

    def __init__(self):
        self.schemas = self._load_contract_schemas()
        self.security_patterns = self._load_security_patterns()

    def _load_contract_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for different contract types"""

        return {
            "evolution_contract": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["contract_id", "version", "evolution_rules", "safety_constraints"],
                "properties": {
                    "contract_id": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9_-]{8,64}$"
                    },
                    "version": {
                        "type": "string",
                        "pattern": "^\\d+\\.\\d+\\.\\d+$"
                    },
                    "description": {
                        "type": "string",
                        "maxLength": 500
                    },
                    "evolution_rules": {
                        "type": "object",
                        "required": ["max_iterations", "fitness_threshold"],
                        "properties": {
                            "max_iterations": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10000
                            },
                            "fitness_threshold": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            },
                            "allowed_operations": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["optimize", "refactor", "add_feature", "remove_code", "restructure"]
                                }
                            }
                        }
                    },
                    "safety_constraints": {
                        "type": "object",
                        "required": ["rollback_enabled", "validation_required"],
                        "properties": {
                            "rollback_enabled": {"type": "boolean"},
                            "validation_required": {"type": "boolean"},
                            "max_execution_time": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 3600
                            },
                            "resource_limits": {
                                "type": "object",
                                "properties": {
                                    "max_memory_mb": {"type": "integer", "minimum": 1, "maximum": 8192},
                                    "max_cpu_percent": {"type": "integer", "minimum": 1, "maximum": 100},
                                    "max_file_operations": {"type": "integer", "minimum": 1, "maximum": 1000}
                                }
                            }
                        }
                    },
                    "authorized_users": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^[a-zA-Z0-9_-]{3,32}$"
                        },
                        "maxItems": 10
                    },
                    "audit_trail": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {"type": "number"},
                                "user": {"type": "string"},
                                "action": {"type": "string"}
                            }
                        }
                    }
                }
            },

            "fitness_function": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["name", "type", "parameters"],
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9_-]{3,50}$"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["performance", "security", "maintainability", "functionality"]
                    },
                    "parameters": {
                        "type": "object",
                        "additionalProperties": {
                            "type": ["string", "number", "boolean"]
                        }
                    },
                    "weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }
        }

    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load security patterns to detect"""

        return {
            "dangerous_functions": [
                "__import__", "eval", "exec", "compile",
                "open", "file", "input", "raw_input",
                "globals", "locals", "vars", "dir",
                "getattr", "setattr", "hasattr", "delattr",
                "importlib", "imp", "sys.modules"
            ],
            "dangerous_modules": [
                "os", "subprocess", "sys", "shutil",
                "pickle", "shelve", "marshal",
                "socket", "urllib", "urllib2", "urllib3",
                "requests", "httplib", "httplib2"
            ],
            "suspicious_patterns": [
                r"\\x[0-9a-fA-F]{2}",  # Hex escapes
                r"\\u[0-9a-fA-F]{4}",  # Unicode escapes
                r"<script[^>]*>.*?</script>",  # Script tags
                r"javascript:",  # JavaScript URLs
                r"data:",  # Data URLs
                r"vbscript:",  # VBScript
                r"on\w+\s*=",  # Event handlers
            ],
            "privilege_escalation": [
                "sudo", "su", "runas", "administrator",
                "root", "wheel", "admin",
                "chmod 777", "chown root",
                "setuid", "setgid"
            ]
        }

    def validate_contract(self, contract: Dict[str, Any],
                         contract_type: str = "evolution_contract") -> ValidationResult:
        """Validate evolution contract against schema and security rules"""

        result = ValidationResult()

        # Step 1: Schema validation
        schema_result = self._validate_schema(contract, contract_type)
        result.errors.extend(schema_result.get('errors', []))
        result.warnings.extend(schema_result.get('warnings', []))

        # Step 2: Security analysis
        security_analysis = self._analyze_security(contract)
        result.security_score = security_analysis.overall_risk_score

        # Add security issues as errors/warnings
        for risk in security_analysis.injection_risks:
            result.errors.append({
                'type': 'security',
                'severity': 'CRITICAL',
                'field': 'contract',
                'message': f'Injection risk: {risk}',
                'suggestion': 'Remove dangerous code patterns'
            })

        for pattern in security_analysis.unsafe_patterns:
            result.warnings.append({
                'type': 'security',
                'severity': 'HIGH',
                'field': 'contract',
                'message': f'Unsafe pattern: {pattern}',
                'suggestion': 'Review and sanitize input'
            })

        # Step 3: Semantic validation
        semantic_issues = self._validate_semantics(contract, contract_type)
        result.errors.extend(semantic_issues.get('errors', []))
        result.warnings.extend(semantic_issues.get('warnings', []))

        # Step 4: Business rule validation
        business_issues = self._validate_business_rules(contract)
        result.errors.extend(business_issues.get('errors', []))
        result.warnings.extend(business_issues.get('warnings', []))

        # Overall validity
        critical_errors = [e for e in result.errors if e.get('severity') == 'CRITICAL']
        result.is_valid = len(critical_errors) == 0

        # Calculate final security score
        if result.is_valid:
            result.security_score = max(0.0, 1.0 - (len(result.warnings) * 0.1) - (len(result.errors) * 0.2))

        return result

    def _validate_schema(self, contract: Dict[str, Any], contract_type: str) -> Dict[str, Any]:
        """Validate contract against JSON schema"""

        result = {'errors': [], 'warnings': []}

        schema = self.schemas.get(contract_type)
        if not schema:
            result['errors'].append({
                'type': 'schema',
                'severity': 'CRITICAL',
                'field': 'contract_type',
                'message': f'Unknown contract type: {contract_type}',
                'suggestion': f'Use one of: {list(self.schemas.keys())}'
            })
            return result

        try:
            jsonschema.validate(contract, schema)
        except jsonschema.ValidationError as e:
            result['errors'].append({
                'type': 'schema',
                'severity': 'HIGH',
                'field': e.absolute_path[0] if e.absolute_path else 'root',
                'message': e.message,
                'suggestion': 'Fix contract structure to match schema'
            })
        except jsonschema.SchemaError as e:
            result['errors'].append({
                'type': 'schema',
                'severity': 'CRITICAL',
                'field': 'schema',
                'message': f'Schema error: {e.message}',
                'suggestion': 'Contact system administrator'
            })

        return result

    def _analyze_security(self, contract: Dict[str, Any]) -> SecurityAnalysis:
        """Analyze contract for security vulnerabilities"""

        analysis = SecurityAnalysis()
        contract_str = json.dumps(contract, sort_keys=True)

        # Check for dangerous functions
        for func in self.security_patterns['dangerous_functions']:
            if func in contract_str:
                analysis.injection_risks.append(f"Potentially dangerous function: {func}")

        # Check for dangerous modules
        for module in self.security_patterns['dangerous_modules']:
            if f'"{module}"' in contract_str or f"'{module}'" in contract_str:
                analysis.injection_risks.append(f"Potentially dangerous module: {module}")

        # Check for suspicious patterns
        for pattern in self.security_patterns['suspicious_patterns']:
            if re.search(pattern, contract_str, re.IGNORECASE):
                analysis.unsafe_patterns.append(f"Suspicious pattern detected: {pattern}")

        # Check for privilege escalation
        for priv in self.security_patterns['privilege_escalation']:
            if priv.lower() in contract_str.lower():
                analysis.privilege_escalation_risks.append(f"Privilege escalation risk: {priv}")

        # Check for data exposure risks
        if 'password' in contract_str.lower() or 'secret' in contract_str.lower():
            if not self._is_properly_encrypted(contract):
                analysis.data_exposure_risks.append("Potential exposure of sensitive data")

        # Calculate overall risk score
        risk_factors = (
            len(analysis.injection_risks) * 0.4 +
            len(analysis.unsafe_patterns) * 0.3 +
            len(analysis.privilege_escalation_risks) * 0.5 +
            len(analysis.data_exposure_risks) * 0.4
        )

        analysis.overall_risk_score = min(1.0, risk_factors)

        return analysis

    def _validate_semantics(self, contract: Dict[str, Any], contract_type: str) -> Dict[str, Any]:
        """Validate semantic correctness of contract"""

        result = {'errors': [], 'warnings': []}

        if contract_type == "evolution_contract":
            # Check semantic relationships
            rules = contract.get('evolution_rules', {})
            constraints = contract.get('safety_constraints', {})

            # Fitness threshold should be reasonable
            fitness_threshold = rules.get('fitness_threshold', 0.5)
            if fitness_threshold < 0.1:
                result['warnings'].append({
                    'type': 'semantic',
                    'severity': 'MEDIUM',
                    'field': 'fitness_threshold',
                    'message': 'Very low fitness threshold may allow poor solutions',
                    'suggestion': 'Consider increasing fitness threshold'
                })

            # Max iterations should be reasonable
            max_iter = rules.get('max_iterations', 100)
            if max_iter > 5000:
                result['warnings'].append({
                    'type': 'semantic',
                    'severity': 'MEDIUM',
                    'field': 'max_iterations',
                    'message': 'Very high iteration limit may cause performance issues',
                    'suggestion': 'Consider reducing max_iterations'
                })

            # Safety constraints should be consistent
            if constraints.get('rollback_enabled') and not constraints.get('validation_required'):
                result['warnings'].append({
                    'type': 'semantic',
                    'severity': 'LOW',
                    'field': 'safety_constraints',
                    'message': 'Rollback enabled but validation disabled',
                    'suggestion': 'Enable validation for safer rollbacks'
                })

        return result

    def _validate_business_rules(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Validate business rules"""

        result = {'errors': [], 'warnings': []}

        # Check version format
        version = contract.get('version', '')
        if not re.match(r'^\d+\.\d+\.\d+$', version):
            result['errors'].append({
                'type': 'business',
                'severity': 'MEDIUM',
                'field': 'version',
                'message': 'Invalid version format (should be x.y.z)',
                'suggestion': 'Use semantic versioning (e.g., 1.0.0)'
            })

        # Check contract ID uniqueness (would need database lookup in real implementation)
        contract_id = contract.get('contract_id', '')
        if len(contract_id) < 8:
            result['warnings'].append({
                'type': 'business',
                'severity': 'LOW',
                'field': 'contract_id',
                'message': 'Contract ID is very short',
                'suggestion': 'Use longer, more unique contract IDs'
            })

        # Check authorized users
        authorized_users = contract.get('authorized_users', [])
        if len(authorized_users) == 0:
            result['warnings'].append({
                'type': 'business',
                'severity': 'MEDIUM',
                'field': 'authorized_users',
                'message': 'No authorized users specified',
                'suggestion': 'Specify authorized users for access control'
            })

        return result

    def _is_properly_encrypted(self, contract: Dict[str, Any]) -> bool:
        """Check if sensitive data is properly encrypted"""
        # This would check for encryption markers, hashes, etc.
        # Simplified implementation
        contract_str = json.dumps(contract)
        return 'encrypted' in contract_str.lower() or 'hash' in contract_str.lower()

    def sanitize_contract(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize contract by removing dangerous content"""

        sanitized = json.loads(json.dumps(contract))  # Deep copy

        # Remove dangerous keys
        dangerous_keys = ['exec', 'eval', 'import', 'code', 'script']

        def remove_dangerous(obj):
            if isinstance(obj, dict):
                return {k: remove_dangerous(v) for k, v in obj.items()
                       if k.lower() not in dangerous_keys}
            elif isinstance(obj, list):
                return [remove_dangerous(item) for item in obj]
            else:
                return obj

        return remove_dangerous(sanitized)

    def generate_contract_hash(self, contract: Dict[str, Any]) -> str:
        """Generate secure hash of contract for integrity checking"""

        # Create canonical JSON representation
        canonical = json.dumps(contract, sort_keys=True, separators=(',', ':'))

        # Generate SHA-256 hash
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def verify_contract_integrity(self, contract: Dict[str, Any], expected_hash: str) -> bool:
        """Verify contract integrity using hash"""

        actual_hash = self.generate_contract_hash(contract)
        return hmac.compare_digest(actual_hash, expected_hash)

class SecureContractLoader:
    """Secure loader for evolution contracts"""

    def __init__(self, validator: EvolutionContractValidator):
        self.validator = validator
        self.loaded_contracts: Dict[str, Dict[str, Any]] = {}
        self.contract_hashes: Dict[str, str] = {}

    def load_contract(self, contract_path: str) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """Load and validate contract securely"""

        errors = []

        # Check file existence and permissions
        if not os.path.exists(contract_path):
            errors.append(f"Contract file does not exist: {contract_path}")
            return False, None, errors

        if not os.access(contract_path, os.R_OK):
            errors.append(f"Cannot read contract file: {contract_path}")
            return False, None, errors

        # Check file size (prevent DoS)
        file_size = os.path.getsize(contract_path)
        if file_size > 1024 * 1024:  # 1MB limit
            errors.append(f"Contract file too large: {file_size} bytes")
            return False, None, errors

        try:
            with open(contract_path, 'r', encoding='utf-8') as f:
                contract_data = json.load(f)

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in contract: {e}")
            return False, None, errors
        except UnicodeDecodeError as e:
            errors.append(f"Invalid encoding in contract: {e}")
            return False, None, errors

        # Validate contract
        validation_result = self.validator.validate_contract(contract_data)

        if not validation_result.is_valid:
            for error in validation_result.errors:
                errors.append(f"Validation error: {error['message']}")

        if validation_result.security_score < 0.7:
            errors.append(f"Security score too low: {validation_result.security_score:.2f}")

        if errors:
            return False, None, errors

        # Generate and store hash for integrity
        contract_hash = self.validator.generate_contract_hash(contract_data)
        contract_id = contract_data.get('contract_id', contract_path)

        self.loaded_contracts[contract_id] = contract_data
        self.contract_hashes[contract_id] = contract_hash

        return True, contract_data, []

    def get_contract(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get loaded contract with integrity verification"""

        if contract_id not in self.loaded_contracts:
            return None

        contract = self.loaded_contracts[contract_id]
        expected_hash = self.contract_hashes[contract_id]

        if not self.validator.verify_contract_integrity(contract, expected_hash):
            print(f"WARNING: Contract {contract_id} integrity check failed!")
            return None

        return contract

def main():
    """CLI interface for contract validation"""
    import argparse

    parser = argparse.ArgumentParser(description="Contract Validation System")
    parser.add_argument("contract_file", help="Path to contract file to validate")
    parser.add_argument("--type", default="evolution_contract",
                       choices=["evolution_contract", "fitness_function"],
                       help="Contract type")
    parser.add_argument("--sanitize", action="store_true",
                       help="Output sanitized version of contract")
    parser.add_argument("--hash", action="store_true",
                       help="Generate and display contract hash")

    args = parser.parse_args()

    validator = EvolutionContractValidator()
    loader = SecureContractLoader(validator)

    print(f"🔍 VALIDATING CONTRACT: {args.contract_file}")
    print(f"Contract Type: {args.type}")
    print("=" * 50)

    # Load and validate contract
    success, contract, errors = loader.load_contract(args.contract_file)

    if not success:
        print("❌ VALIDATION FAILED")
        for error in errors:
            print(f"  • {error}")
        return 1

    print("✅ VALIDATION PASSED")

    if contract:
        # Display validation details
        validation_result = validator.validate_contract(contract, args.type)

        print("\n📊 VALIDATION SUMMARY")
        print(f"  Schema Valid: ✅")
        print(f"  Security Score: {validation_result.security_score:.2f}/1.00")
        print(f"  Warnings: {len(validation_result.warnings)}")
        print(f"  Errors: {len(validation_result.errors)}")

        if validation_result.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in validation_result.warnings[:5]:  # Show first 5
                print(f"  • {warning['message']}")

        if args.hash:
            contract_hash = validator.generate_contract_hash(contract)
            print("\n🔐 CONTRACT HASH:")
            print(f"  SHA-256: {contract_hash}")

        if args.sanitize:
            sanitized = validator.sanitize_contract(contract)
            print("\n🧹 SANITIZED CONTRACT:")
            print(json.dumps(sanitized, indent=2)[:500] + "..." if len(json.dumps(sanitized)) > 500 else json.dumps(sanitized, indent=2))

    print("\n🎉 Contract validation completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
