#!/usr/bin/env python3
"""
ULTRA CRITIC SWARM - 13-Agent Code Review System
================================================

Simulates the ultra-harsh critic swarm analysis for consciousness computing code.
"""

import re
import ast
import os
from pathlib import Path
from datetime import datetime
import json


class UltraCriticSwarm:
    """Ultra-harsh critic swarm with 13 specialized AI agents"""

    def __init__(self):
        self.critics = {
            'devil_advocate': DevilAdvocate(),
            'stress_tester': StressTester(),
            'edge_case_hunter': EdgeCaseHunter(),
            'logic_destroyer': LogicDestroyer(),
            'ui_flow_breaker': UIFlowBreaker(),
            'security_paranoid': SecurityParanoid(),
            'performance_nazi': PerformanceNazi(),
            'memory_leak_hunter': MemoryLeakHunter(),
            'race_condition_finder': RaceConditionFinder(),
            'input_fuzzer': InputFuzzer(),
            'dependency_skeptic': DependencySkeptic(),
            'error_path_explorer': ErrorPathExplorer(),
            'assumption_challenger': AssumptionChallenger()
        }

    def review_file(self, file_path):
        """Review a single file with all 13 critics"""
        print(f"🔍 Ultra Critic Swarm analyzing: {file_path}")

        if not Path(file_path).exists():
            return {
                'target': file_path,
                'overall_score': 0,
                'verdict': 'CATASTROPHIC',
                'findings': [{'severity': 'CRITICAL', 'message': 'File does not exist'}]
            }

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        findings = []

        # Run all critics
        for critic_name, critic in self.critics.items():
            try:
                critic_findings = critic.analyze(content, file_path)
                findings.extend(critic_findings)
            except Exception as e:
                findings.append({
                    'severity': 'HIGH',
                    'category': critic_name,
                    'message': f'Critic {critic_name} failed: {e}',
                    'exploit_scenario': 'Analysis failure indicates code complexity issues'
                })

        # Calculate overall score
        score = self._calculate_overall_score(findings)

        # Determine verdict
        verdict = self._determine_verdict(score)

        report = {
            'target': file_path,
            'timestamp': datetime.now().isoformat(),
            'overall_score': score,
            'verdict': verdict,
            'total_findings': len(findings),
            'findings_by_severity': self._group_findings_by_severity(findings),
            'findings': findings[:50],  # Limit output
            'critics_executed': len(self.critics)
        }

        return report

    def _calculate_overall_score(self, findings):
        """Calculate overall score from findings"""
        if not findings:
            return 100

        # Weight by severity
        severity_weights = {
            'CRITICAL': 15,
            'HIGH': 10,
            'MEDIUM': 5,
            'LOW': 2,
            'INFO': 1
        }

        total_penalty = 0
        for finding in findings:
            severity = finding.get('severity', 'INFO')
            total_penalty += severity_weights.get(severity, 1)

        # Cap at 100 points deduction
        score = max(0, 100 - min(total_penalty, 100))
        return round(score, 1)

    def _determine_verdict(self, score):
        """Determine verdict based on score"""
        if score >= 90:
            return 'ACCEPTABLE'
        elif score >= 70:
            return 'NEEDS WORK'
        elif score >= 50:
            return 'POOR'
        elif score >= 25:
            return 'CRITICAL'
        else:
            return 'CATASTROPHIC'

    def _group_findings_by_severity(self, findings):
        """Group findings by severity"""
        groups = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0}
        for finding in findings:
            severity = finding.get('severity', 'INFO')
            groups[severity] += 1
        return groups

    def format_report_text(self, report):
        """Format report as text"""
        output = []
        output.append("=" * 80)
        output.append(f"ULTRA CRITIC SWARM REPORT")
        output.append("=" * 80)
        output.append(f"Target: {report['target']}")
        output.append(f"Timestamp: {report['timestamp']}")
        output.append("")
        output.append(f"OVERALL SCORE: {report['overall_score']}/100")
        output.append(f"VERDICT: {report['verdict']}")
        output.append("")
        output.append("FINDINGS BY SEVERITY:")
        for severity, count in report['findings_by_severity'].items():
            if count > 0:
                output.append(f"  {severity}: {count}")
        output.append("")
        output.append("TOP FINDINGS:")

        # Show top 10 findings
        sorted_findings = sorted(report['findings'],
                               key=lambda x: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'].index(x.get('severity', 'INFO')))

        for i, finding in enumerate(sorted_findings[:10], 1):
            output.append(f"{i}. [{finding.get('severity', 'INFO')}] {finding.get('category', 'unknown')}")
            output.append(f"   {finding.get('message', 'No message')}")
            if finding.get('exploit_scenario'):
                output.append(f"   EXPLOIT: {finding['exploit_scenario']}")
            output.append("")

        output.append("=" * 80)
        return "\n".join(output)


# Critic Agent Classes
class BaseCritic:
    """Base critic class"""
    def __init__(self, name):
        self.name = name

    def analyze(self, content, file_path):
        """Analyze content and return findings"""
        return []


class DevilAdvocate(BaseCritic):
    """Assumes everything is wrong, argues against all decisions"""

    def __init__(self):
        super().__init__('devil_advocate')

    def analyze(self, content, file_path):
        findings = []

        # Check for overly complex logic
        if len(content.split('\n')) > 200:
            findings.append({
                'severity': 'HIGH',
                'category': 'devil_advocate',
                'message': 'File is excessively long - complexity suggests poor design decisions',
                'exploit_scenario': 'Maintenance nightmare, bug harbor'
            })

        # Check for imports
        if 'import' in content:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'devil_advocate',
                'message': 'External dependencies are a liability - why not reinvent everything?',
                'exploit_scenario': 'Supply chain attacks through dependencies'
            })

        # Check for classes
        class_count = content.count('class ')
        if class_count > 5:
            findings.append({
                'severity': 'HIGH',
                'category': 'devil_advocate',
                'message': f'{class_count} classes? This inheritance hierarchy is probably wrong',
                'exploit_scenario': 'Complex inheritance leads to method resolution disasters'
            })

        return findings


class StressTester(BaseCritic):
    """Breaks things under extreme load"""

    def __init__(self):
        super().__init__('stress_tester')

    def analyze(self, content, file_path):
        findings = []

        # Check for loops without bounds
        if 'while True' in content or 'for _ in ' in content:
            findings.append({
                'severity': 'CRITICAL',
                'category': 'stress_tester',
                'message': 'Unbounded loops will crash under high load',
                'exploit_scenario': 'Infinite loop DoS attack with 1M concurrent requests'
            })

        # Check for memory allocations
        if 'list(' in content or 'dict(' in content:
            findings.append({
                'severity': 'HIGH',
                'category': 'stress_tester',
                'message': 'Memory allocations without limits will cause OOM under load',
                'exploit_scenario': 'Memory exhaustion with 1TB input files'
            })

        # Check for recursion
        if 'def ' in content and content.count('return') < content.count('def '):
            findings.append({
                'severity': 'CRITICAL',
                'category': 'stress_tester',
                'message': 'Potential infinite recursion will stack overflow',
                'exploit_scenario': 'Stack overflow with deep nested inputs'
            })

        return findings


class EdgeCaseHunter(BaseCritic):
    """Empty strings, null bytes, unicode bombs, MAX_INT"""

    def __init__(self):
        super().__init__('edge_case_hunter')

    def analyze(self, content, file_path):
        findings = []

        # Check for string handling without validation
        if '.strip()' not in content and ('input(' in content or 'request.' in content):
            findings.append({
                'severity': 'HIGH',
                'category': 'edge_case_hunter',
                'message': 'No input validation - empty strings will crash',
                'exploit_scenario': 'Empty string injection causes crashes'
            })

        # Check for encoding handling
        if 'encode(' not in content and 'decode(' not in content and 'utf-8' not in content:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'edge_case_hunter',
                'message': 'No explicit encoding handling - Unicode bombs will break',
                'exploit_scenario': 'Unicode normalization attacks'
            })

        # Check for integer overflow potential
        if 'int(' in content and 'max(' not in content:
            findings.append({
                'severity': 'HIGH',
                'category': 'edge_case_hunter',
                'message': 'Integer conversion without bounds checking',
                'exploit_scenario': 'MAX_INT overflow crashes'
            })

        # Check for None/null handling
        if 'if ' in content and 'is None' not in content and 'None' in content:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'edge_case_hunter',
                'message': 'Potential null pointer dereferences',
                'exploit_scenario': 'None/null value crashes'
            })

        return findings


class LogicDestroyer(BaseCritic):
    """Finds contradictions, impossible states, broken invariants"""

    def __init__(self):
        super().__init__('logic_destroyer')

    def analyze(self, content, file_path):
        findings = []

        # Check for contradictory conditions
        if 'if' in content and 'else' in content:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'logic_destroyer',
                'message': 'Conditional logic may have unreachable code paths',
                'exploit_scenario': 'Logic contradictions lead to impossible states'
            })

        # Check for circular dependencies (imports)
        lines = content.split('\n')
        imports = [line for line in lines if line.strip().startswith('from') or line.strip().startswith('import')]
        if len(imports) > 3:
            findings.append({
                'severity': 'HIGH',
                'category': 'logic_destroyer',
                'message': 'Multiple imports suggest potential circular dependency issues',
                'exploit_scenario': 'Import cycles cause initialization failures'
            })

        # Check for inconsistent error handling
        if 'try:' in content and 'except:' not in content:
            findings.append({
                'severity': 'HIGH',
                'category': 'logic_destroyer',
                'message': 'Try blocks without except clauses break exception invariants',
                'exploit_scenario': 'Unhandled exceptions break program logic'
            })

        return findings


class SecurityParanoid(BaseCritic):
    """Injection, auth bypass, hardcoded secrets, supply chain"""

    def __init__(self):
        super().__init__('security_paranoid')

    def analyze(self, content, file_path):
        findings = []

        # Check for SQL injection potential
        if ('execute(' in content or 'query(' in content) and '%' not in content:
            findings.append({
                'severity': 'CRITICAL',
                'category': 'security_paranoid',
                'message': 'Potential SQL injection - no parameterized queries',
                'exploit_scenario': 'SQL injection data exfiltration'
            })

        # Check for hardcoded secrets
        secret_patterns = ['password', 'secret', 'key', 'token']
        for pattern in secret_patterns:
            if pattern in content.lower() and ('=' in content or ':' in content):
                findings.append({
                    'severity': 'CRITICAL',
                    'category': 'security_paranoid',
                    'message': f'Potential hardcoded {pattern} detected',
                    'exploit_scenario': 'Credential theft from source code'
                })

        # Check for command injection
        if 'subprocess' in content or 'os.system' in content:
            findings.append({
                'severity': 'HIGH',
                'category': 'security_paranoid',
                'message': 'System command execution without sanitization',
                'exploit_scenario': 'Command injection remote code execution'
            })

        # Check for authentication bypass
        if 'admin' in content.lower() and '==' in content:
            findings.append({
                'severity': 'CRITICAL',
                'category': 'security_paranoid',
                'message': 'Simple equality checks for admin access',
                'exploit_scenario': 'Authentication bypass with type confusion'
            })

        return findings


class PerformanceNazi(BaseCritic):
    """O(n^2) loops, allocations, cache misses"""

    def __init__(self):
        super().__init__('performance_nazi')

    def analyze(self, content, file_path):
        findings = []

        # Check for nested loops
        if 'for ' in content and content.count('for ') > 1:
            findings.append({
                'severity': 'HIGH',
                'category': 'performance_nazi',
                'message': 'Nested loops suggest O(n^2) complexity',
                'exploit_scenario': 'Performance degradation with large datasets'
            })

        # Check for inefficient list operations
        if '.append(' in content and 'for ' in content:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'performance_nazi',
                'message': 'List append in loop may cause excessive reallocations',
                'exploit_scenario': 'Memory fragmentation and cache misses'
            })

        # Check for string concatenation in loops
        if '+' in content and 'for ' in content and '"' in content:
            findings.append({
                'severity': 'HIGH',
                'category': 'performance_nazi',
                'message': 'String concatenation in loops creates O(n^2) behavior',
                'exploit_scenario': 'Exponential memory usage with iterations'
            })

        return findings


class MemoryLeakHunter(BaseCritic):
    """Unclosed files, event listeners, circular refs"""

    def __init__(self):
        super().__init__('memory_leak_hunter')

    def analyze(self, content, file_path):
        findings = []

        # Check for file operations without context managers
        if 'open(' in content and 'with ' not in content:
            findings.append({
                'severity': 'HIGH',
                'category': 'memory_leak_hunter',
                'message': 'File operations without context managers will leak handles',
                'exploit_scenario': 'File handle exhaustion DoS'
            })

        # Check for event listeners without cleanup
        if 'addEventListener' in content or 'on(' in content:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'memory_leak_hunter',
                'message': 'Event listeners may not be cleaned up',
                'exploit_scenario': 'Memory leaks from accumulated listeners'
            })

        # Check for circular references potential
        if 'self.' in content and '__del__' not in content:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'memory_leak_hunter',
                'message': 'Self-references without __del__ may create circular refs',
                'exploit_scenario': 'Memory leaks from circular references'
            })

        return findings


class InputFuzzer(BaseCritic):
    """Chaos payloads, type confusion, encoding attacks"""

    def __init__(self):
        super().__init__('input_fuzzer')

    def analyze(self, content, file_path):
        findings = []

        # Check for type conversion without validation
        if 'int(' in content and 'try:' not in content:
            findings.append({
                'severity': 'HIGH',
                'category': 'input_fuzzer',
                'message': 'Integer conversion without exception handling',
                'exploit_scenario': 'Type confusion attacks with malformed input'
            })

        # Check for JSON parsing
        if 'json.loads' in content and 'try:' not in content:
            findings.append({
                'severity': 'HIGH',
                'category': 'input_fuzzer',
                'message': 'JSON parsing without error handling',
                'exploit_scenario': 'JSON parsing attacks with malformed payloads'
            })

        # Check for regex without timeout
        if 're.' in content and 'timeout' not in content:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'input_fuzzer',
                'message': 'Regex operations without timeout limits',
                'exploit_scenario': 'ReDoS attacks with catastrophic backtracking'
            })

        return findings


class AssumptionChallenger(BaseCritic):
    """Hidden assumptions, implicit contracts"""

    def __init__(self):
        super().__init__('assumption_challenger')

    def analyze(self, content, file_path):
        findings = []

        # Check for magic numbers
        magic_numbers = re.findall(r'\b\d{2,}\b', content)
        if len(magic_numbers) > 5:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'assumption_challenger',
                'message': f'{len(magic_numbers)} magic numbers found - hidden assumptions',
                'exploit_scenario': 'Magic number changes break implicit contracts'
            })

        # Check for TODO comments
        if 'TODO' in content or 'FIXME' in content:
            findings.append({
                'severity': 'LOW',
                'category': 'assumption_challenger',
                'message': 'TODO/FIXME comments indicate incomplete assumptions',
                'exploit_scenario': 'Incomplete features lead to unexpected behavior'
            })

        # Check for single-point-of-failure functions
        if 'def ' in content and len(content.split('def ')) > 10:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'assumption_challenger',
                'message': 'Many functions suggest single-responsibility violations',
                'exploit_scenario': 'God functions break encapsulation assumptions'
            })

        return findings


# Placeholder classes for remaining critics
class UIFlowBreaker(BaseCritic):
    def __init__(self):
        super().__init__('ui_flow_breaker')

    def analyze(self, content, file_path):
        return [{
            'severity': 'LOW',
            'category': 'ui_flow_breaker',
            'message': 'No UI components detected for flow analysis',
            'exploit_scenario': 'UI flow issues would cause user experience problems'
        }]


class RaceConditionFinder(BaseCritic):
    def __init__(self):
        super().__init__('race_condition_finder')

    def analyze(self, content, file_path):
        if 'threading' in content or 'asyncio' in content:
            return [{
                'severity': 'HIGH',
                'category': 'race_condition_finder',
                'message': 'Concurrent code without proper synchronization',
                'exploit_scenario': 'Race conditions in multi-threaded operations'
            }]
        return []


class DependencySkeptic(BaseCritic):
    def __init__(self):
        super().__init__('dependency_skeptic')

    def analyze(self, content, file_path):
        if 'requirements.txt' in str(file_path) or 'pip install' in content:
            return [{
                'severity': 'MEDIUM',
                'category': 'dependency_skeptic',
                'message': 'External dependencies may have CVEs',
                'exploit_scenario': 'Supply chain attacks through vulnerable dependencies'
            }]
        return []


class ErrorPathExplorer(BaseCritic):
    def __init__(self):
        super().__init__('error_path_explorer')

    def analyze(self, content, file_path):
        if 'except:' in content and 'pass' in content:
            return [{
                'severity': 'HIGH',
                'category': 'error_path_explorer',
                'message': 'Bare except clauses swallow all errors',
                'exploit_scenario': 'Silent failure masking critical errors'
            }]
        return []


def main():
    """Run ultra critic swarm on consciousness security fixes"""
    swarm = UltraCriticSwarm()

    target_file = "consciousness_security_fixes.py"
    print(f"🎯 Executing Ultra Critic Swarm on: {target_file}")
    print("=" * 60)

    report = swarm.review_file(target_file)

    print(swarm.format_report_text(report))

    print("\n" + "=" * 80)
    print("ULTRA CRITIC SWARM EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
