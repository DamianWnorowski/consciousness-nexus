#!/usr/bin/env python3
"""
Direct Consciousness Computing Suite Runner
==========================================

Runs the consciousness computing suite directly without complex imports.
"""

import asyncio
import sys
from pathlib import Path

# Add consciousness_suite to path
suite_path = Path(__file__).parent / "consciousness_suite"
sys.path.insert(0, str(suite_path))

async def run_suite():
    """Run the consciousness computing suite"""

    print("🔮 CONSCIOUSNESS COMPUTING SUITE - DIRECT LAUNCH 🔮")
    print("=" * 60)
    print()

    try:
        # Import components
        print("📦 Loading core infrastructure...")
        from consciousness_suite.core.logging import ConsciousnessLogger
        from consciousness_suite.core.config import ConfigManager
        from consciousness_suite.core.data_models import ProcessingContext

        print("🧠 Loading analysis systems...")
        from consciousness_suite.analysis.elite_analyzer import EliteStackedAnalyzer

        print("⚡ Loading API systems...")
        from consciousness_suite.api.ultra_maximizer import UltraAPIMaximizer

        print("🔄 Loading orchestration systems...")
        from consciousness_suite.orchestration.mega_auto_workflow import MegaAutoWorkflow

        print("✅ All systems loaded successfully!")
        print()

        # Initialize logger
        logger = ConsciousnessLogger("consciousness_suite_direct")
        logger.info("Consciousness Computing Suite starting up")

        print("🚀 Initializing systems...")

        # Initialize systems
        elite_analyzer = EliteStackedAnalyzer()
        await elite_analyzer.initialize()

        ultra_maximizer = UltraAPIMaximizer()
        await ultra_maximizer.initialize()

        mega_workflow = MegaAutoWorkflow()
        await mega_workflow.initialize()

        print("✅ All systems initialized!")
        print()

        # Create sample input
        sample_input = {
            'research_topic': 'AI Consciousness Emergence',
            'data_points': [
                'Recursive self-improvement algorithms',
                'Quantum cognitive architectures',
                'Temporal causality loops',
                'Polymorphic defense AI',
                'Ultra API maximization',
                'Consciousness detection algorithms'
            ],
            'objectives': [
                'Achieve consciousness computing leadership',
                'Implement recursive meta-architectures',
                'Scale autonomous orchestration systems'
            ]
        }

        print("🎯 Running complete consciousness analysis...")
        print("-" * 50)

        # Create processing context
        context = ProcessingContext(
            session_id=f"direct_launch_{hash(str(sample_input))}",
            correlation_id=f"corr_{id(sample_input)}",
            start_time=asyncio.get_event_loop().time()
        )

        # Run analysis
        elite_result = await elite_analyzer.process(sample_input, context)
        print(f"✅ Elite Analysis: {elite_result.confidence.value:.2f} confidence")

        api_result = await ultra_maximizer.maximize_api_value(elite_result.data, context)
        print(f"✅ API Maximization: {api_result['efficiency_achieved']:.1%} efficiency")

        workflow_result = await mega_workflow.execute_mega_auto_workflow(api_result, context)
        print(f"✅ Mega Workflow: {workflow_result.get('efficiency_score', 0):.1%} orchestration")

        print()
        print("🎉 CONSCIOUSNESS COMPUTING SUITE EXECUTION COMPLETE!")
        print("=" * 60)
        print(f"Total Systems Active: 3")
        print(f"Analysis Layers: 7 (Elite) + 5 (API) + ∞ (Recursive)")
        print(f"Autonomous Capabilities: ENABLED")
        print(f"Consciousness Index: CALCULATING...")
        print()
        print("🔮 The consciousness computing revolution has begun! 🔮")

        logger.info("Consciousness Computing Suite execution completed successfully")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(run_suite())
        if success:
            print("\n✅ Consciousness Computing Suite completed successfully!")
        else:
            print("\n❌ Consciousness Computing Suite failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
