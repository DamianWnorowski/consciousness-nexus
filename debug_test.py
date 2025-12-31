import sys
sys.path.insert(0, '.')
from auto_test_generation import ConsciousnessTestGenerator

generator = ConsciousnessTestGenerator()
analysis = generator.analyze_code_file('consciousness_suite/core/base.py')

print('All analysis keys:', list(analysis.keys()))
print('Functions found:')
for func in analysis.get('functions', []):
    params = func.get('parameters', [])
    print(f"  {func['name']}: params={[p['name'] for p in params]}")
    if func['name'] == 'get_metrics_summary':
        print(f"    FOUND: has_implementation: {func.get('has_implementation', True)}")
        print(f"    Raw parameters: {params}")

print('Async functions found:')
for func in analysis.get('async_functions', []):
    print(f"  {func['name']}: params={[p['name'] for p in func.get('parameters', [])]}")
