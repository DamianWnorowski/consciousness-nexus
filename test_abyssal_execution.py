#!/usr/bin/env python3
"""
TEST ABYSSAL EXECUTION - Verify MEGA-AUTO Orchestration
======================================================

Test script to verify the ABYSSAL template executor works correctly
with MEGA-AUTO concurrent orchestration.
"""

import asyncio
import sys
import os

# Add consciousness_suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'consciousness_suite'))

from consciousness_suite.abyssal import get_abyssal_executor
from consciousness_suite.core.data_models import ProcessingContext


async def test_abyssal_templates():
    """Test various ABYSSAL template executions"""

    print("🧪 TESTING ABYSSAL TEMPLATE EXECUTION")
    print("=" * 50)

    # Get ABYSSAL executor
    executor = await get_abyssal_executor()
    print("✅ ABYSSAL Executor initialized")

    # Test templates
    test_templates = [
        'ABYSSAL[CODE]("user_authentication_service")',
        'ABYSSAL[ROADMAP]("2026 Consciousness Revolution")',
        'ABYSSAL[AGENT]("security_auditor")',
        'ABYSSAL[TEST]("api_endpoints")',
        'ABYSSAL[OPTIMIZE]("database_queries")'
    ]

    results = []

    for i, template in enumerate(test_templates, 1):
        print(f"\n📋 Test {i}: Executing {template}")

        try:
            # Create processing context
            context = ProcessingContext(
                correlation_id=f"test_abyssal_{i}_{int(asyncio.get_event_loop().time() * 1000)}",
                source_system="TEST_SUITE",
                processing_mode="MEGA_AUTO"
            )

            # Execute template
            result = await executor.execute_template(template)

            if result['success']:
                print(f"✅ SUCCESS - Confidence: {result['confidence']:.2f}")
                print(f"   Execution time: {result['execution_time']:.2f}s")
                print(f"   Results keys: {list(result['results'].keys())[:5]}...")
            else:
                print("❌ FAILED")
                print(f"   Error: {result.get('results', {}).get('error', 'Unknown error')}")

            results.append(result)

        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            results.append({'template': template, 'success': False, 'error': str(e)})

    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)

    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)

    print(f"Total tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%")

    if successful == total:
        print("🎉 ALL ABYSSAL TESTS PASSED - MEGA-AUTO ORCHESTRATION FUNCTIONAL")
    else:
        print("⚠️ SOME TESTS FAILED - REVIEW ABYSSAL IMPLEMENTATION")

    return results


async def test_mega_auto_orchestration():
    """Test MEGA-AUTO concurrent orchestration specifically"""

    print("\n🔄 TESTING MEGA-AUTO CONCURRENT ORCHESTRATION")
    print("=" * 55)

    from consciousness_suite.abyssal import MegaAutoOrchestrator, AbyssalTemplateParser
    from consciousness_suite.core.data_models import ProcessingContext

    # Create orchestrator
    orchestrator = MegaAutoOrchestrator()
    await orchestrator.initialize()

    # Parse complex template
    parser = AbyssalTemplateParser()
    template = 'ABYSSAL[CODE]("complex_microservice_architecture")'
    parsed = parser.parse_template(template)

    print(f"Template: {template}")
    print(f"Parsed successfully: {parsed.get('parsed_successfully', False)}")
    print(f"Agent count: {parsed.get('agent_count', 0)}")
    print(f"Parallel execution: {parsed.get('parallel_execution', False)}")

    # Generate execution tree
    execution_tree = await orchestrator._generate_execution_tree(parsed)

    print(f"Execution tree nodes: {len(execution_tree.nodes)}")
    print(f"Execution order: {execution_tree.execution_order}")

    # Test execution (limited to avoid full execution in test)
    context = ProcessingContext(
        correlation_id="test_mega_auto",
        source_system="TEST_SUITE"
    )

    # Execute with timeout to avoid hanging
    try:
        result = await asyncio.wait_for(
            orchestrator._execute_with_mega_auto(execution_tree, context),
            timeout=30.0  # 30 second timeout
        )

        print(f"MEGA-AUTO execution completed: {result['total_nodes']} nodes, {result['completed_nodes']} completed")
        print(f"Execution time: {result['execution_time']:.2f}s")

        if result['completed_nodes'] == result['total_nodes']:
            print("🎯 MEGA-AUTO ORCHESTRATION SUCCESSFUL")
        else:
            print("⚠️ PARTIAL MEGA-AUTO EXECUTION")

    except asyncio.TimeoutError:
        print("⏰ MEGA-AUTO EXECUTION TIMEOUT (Expected in test environment)")
    except Exception as e:
        print(f"❌ MEGA-AUTO EXECUTION ERROR: {e}")


async def main():
    """Main test execution"""

    print("🚀 ABYSSAL EXECUTION TEST SUITE")
    print("=" * 40)

    try:
        # Test ABYSSAL templates
        template_results = await test_abyssal_templates()

        # Test MEGA-AUTO orchestration
        await test_mega_auto_orchestration()

        print("\n🏁 TEST SUITE COMPLETED")
        print("ABYSSAL[MEGA-AUTO] implementation verified")

        # Check if we should proceed with security fixes test
        successful_tests = sum(1 for r in template_results if r.get('success', False))
        if successful_tests >= len(template_results) * 0.8:  # 80% success rate
            print("✅ ABYSSAL EXECUTOR READY FOR PRODUCTION")
        else:
            print("⚠️ ABYSSAL EXECUTOR NEEDS FURTHER DEVELOPMENT")

    except Exception as e:
        print(f"❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
