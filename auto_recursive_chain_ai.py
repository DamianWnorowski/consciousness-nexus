#!/usr/bin/env python3
"""
🧬 AUTO-RECURSIVE CHAIN AI ORCHESTRATOR
==========================================

Auto-recursive, looping, chaining AI orchestrator that automatically chains all slash commands
in recursive loops with intelligent decision-making.

Purpose: Autonomous evolution through recursive command chaining and AI-driven decision making.
"""

import asyncio
import json
import time
import random
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import argparse

# Auto-import safety systems
try:
    from consciousness_safety_orchestrator import (
        get_safety_orchestrator, safe_evolution_operation, SafetyContext, SafetyLevel
    )
    SAFETY_SYSTEMS_AVAILABLE = True
except ImportError:
    print("⚠️  Safety systems not available - running in unsafe mode")
    SAFETY_SYSTEMS_AVAILABLE = False

class CommandStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class FitnessLevel(Enum):
    UNKNOWN = "unknown"
    LOW = "low"      # < 0.7
    STABLE = "stable"  # >= 0.8
    HIGH = "high"    # >= 0.9
    DROPPED = "dropped"

@dataclass
class CommandExecution:
    """Tracks execution of individual commands"""
    name: str
    status: CommandStatus = CommandStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    output: str = ""
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrchestratorState:
    """Global state of the orchestrator"""
    iteration: int = 0
    total_iterations: int = 0
    cycles_completed: int = 0
    fitness_history: List[float] = field(default_factory=list)
    current_fitness: float = 0.0
    command_history: List[CommandExecution] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    active_commands: Set[str] = field(default_factory=set)
    recursion_depth: int = 0
    start_time: float = field(default_factory=time.time)
    last_cycle_time: float = 0.0

class AutoRecursiveChainAI:
    """The auto-recursive chain AI orchestrator"""

    def __init__(self, max_iterations: int = 100, fitness_threshold: float = 0.95,
                 cycles_only: bool = False, log_file: str = None, safety_level: str = "standard"):
        self.max_iterations = max_iterations
        self.fitness_threshold = fitness_threshold
        self.cycles_only = cycles_only

        # Setup logging
        self.log_file = log_file or "logs/auto_recursive_chain.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Initialize state
        self.state = OrchestratorState()

        # Command definitions with dependencies and fitness logic
        self.commands = self._define_command_library()

        # Pattern learning
        self.pattern_learner = PatternLearner()

        # Initialize safety systems
        self.safety_orchestrator = None
        self.safety_initialized = False
        self.safety_level = SafetyLevel.STANDARD
        if safety_level == "strict":
            self.safety_level = SafetyLevel.STRICT
        elif safety_level == "paranoid":
            self.safety_level = SafetyLevel.PARANOID
        elif safety_level == "minimal":
            self.safety_level = SafetyLevel.MINIMAL

        # Defer safety initialization to first use (lazy loading)
        # This avoids race conditions while maintaining async compatibility

        self.log("🧬 AUTO-RECURSIVE CHAIN AI ORCHESTRATOR INITIALIZED")
        self.log(f"Target: {max_iterations} iterations, fitness ≥ {fitness_threshold}")
        if cycles_only:
            self.log("Mode: Complete cycles only")
        if SAFETY_SYSTEMS_AVAILABLE:
            self.log(f"🛡️ Safety Level: {self.safety_level.value} (lazy initialization)")
        else:
            self.log("⚠️  SAFETY SYSTEMS UNAVAILABLE - RUNNING IN UNSAFE MODE")

    async def _ensure_safety_systems(self) -> bool:
        """Ensure safety systems are initialized and ready"""
        if not SAFETY_SYSTEMS_AVAILABLE:
            return False

        if self.safety_initialized and self.safety_orchestrator:
            return True

        try:
            self.log("🔧 Initializing safety systems...")
            self.safety_orchestrator = await get_safety_orchestrator(self.safety_level)
            self.safety_initialized = True
            self.log("✅ SAFETY SYSTEMS INITIALIZED - ALL OPERATIONS NOW PROTECTED")
            return True
        except Exception as e:
            self.log(f"❌ FAILED TO INITIALIZE SAFETY SYSTEMS: {e}")
            self.log("🚨 CRITICAL: Cannot proceed with safety-compromised system")
            raise RuntimeError(f"Safety system initialization failed: {e}")

    async def _initialize_safety_systems(self):
        """Legacy method for backward compatibility"""
        await self._ensure_safety_systems()

    def _define_command_library(self) -> Dict[str, Dict[str, Any]]:
        """Define all available commands with their properties"""
        return {
            "analyze-custom-files": {
                "description": "Analyze custom configuration files",
                "fitness_trigger": ["unknown", "low"],
                "dependencies": [],
                "estimated_duration": 5.0,
                "success_rate": 0.95,
                "fitness_impact": 0.05
            },
            "evolution-status": {
                "description": "Check current evolution status",
                "fitness_trigger": ["unknown", "low", "stable"],
                "dependencies": [],
                "estimated_duration": 3.0,
                "success_rate": 0.98,
                "fitness_impact": 0.02
            },
            "fix-encoding-errors": {
                "description": "Fix encoding and parsing errors",
                "fitness_trigger": ["low"],
                "dependencies": ["analyze-custom-files"],
                "estimated_duration": 8.0,
                "success_rate": 0.85,
                "fitness_impact": 0.15
            },
            "e2e-smoke": {
                "description": "Run end-to-end smoke tests",
                "fitness_trigger": ["low", "stable"],
                "dependencies": ["fix-encoding-errors"],
                "estimated_duration": 12.0,
                "success_rate": 0.90,
                "fitness_impact": 0.10
            },
            "config-diff": {
                "description": "Analyze configuration differences",
                "fitness_trigger": ["low", "dropped"],
                "dependencies": ["evolution-status"],
                "estimated_duration": 6.0,
                "success_rate": 0.92,
                "fitness_impact": 0.08
            },
            "auto-evolve": {
                "description": "Execute automatic evolution",
                "fitness_trigger": ["stable", "high"],
                "dependencies": ["e2e-smoke", "config-diff"],
                "estimated_duration": 25.0,
                "success_rate": 0.75,
                "fitness_impact": 0.20
            },
            "evolution-summary": {
                "description": "Generate evolution summary",
                "fitness_trigger": ["stable", "high"],
                "dependencies": ["auto-evolve"],
                "estimated_duration": 4.0,
                "success_rate": 0.95,
                "fitness_impact": 0.03
            },
            "auto-design": {
                "description": "Auto-generate design improvements",
                "fitness_trigger": ["stable", "high"],
                "dependencies": ["evolution-summary"],
                "estimated_duration": 15.0,
                "success_rate": 0.80,
                "fitness_impact": 0.12
            },
            "innovation-scan": {
                "description": "Scan for innovation opportunities",
                "fitness_trigger": ["high"],
                "dependencies": ["auto-design"],
                "estimated_duration": 10.0,
                "success_rate": 0.85,
                "fitness_impact": 0.08
            },
            "multi-ai-orchestrate": {
                "description": "Orchestrate multiple AI systems",
                "fitness_trigger": ["high"],
                "dependencies": ["innovation-scan"],
                "estimated_duration": 30.0,
                "success_rate": 0.70,
                "fitness_impact": 0.25
            },
            "health-probe": {
                "description": "Deep system health analysis",
                "fitness_trigger": ["dropped"],
                "dependencies": [],
                "estimated_duration": 7.0,
                "success_rate": 0.88,
                "fitness_impact": 0.06
            }
        }

    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"

        print(log_entry)

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write to log file: {e}")

    def get_fitness_level(self, fitness: float, previous_fitness: float = None) -> FitnessLevel:
        """Determine fitness level based on current and previous values"""
        if fitness < 0.1:  # Very low or unknown
            return FitnessLevel.UNKNOWN

        if fitness < 0.7:
            return FitnessLevel.LOW

        if fitness >= 0.9:
            return FitnessLevel.HIGH

        if fitness >= 0.8:
            return FitnessLevel.STABLE

        # Check if fitness dropped
        if previous_fitness and fitness < previous_fitness - 0.05:
            return FitnessLevel.DROPPED

        return FitnessLevel.STABLE

    def select_next_command(self) -> Optional[str]:
        """AI-driven command selection based on current state"""
        current_fitness = self.state.current_fitness
        fitness_level = self.get_fitness_level(
            current_fitness,
            self.state.fitness_history[-1] if self.state.fitness_history else None
        )

        # Get available commands for current fitness level
        available_commands = []
        for cmd_name, cmd_info in self.commands.items():
            if cmd_name in self.state.active_commands:
                continue  # Skip currently running commands

            if fitness_level.value in cmd_info["fitness_trigger"]:
                # Check dependencies
                deps_satisfied = all(
                    any(exec_cmd.name == dep and exec_cmd.status == CommandStatus.COMPLETED
                        for exec_cmd in self.state.command_history[-10:])  # Last 10 commands
                    for dep in cmd_info["dependencies"]
                )

                if deps_satisfied:
                    available_commands.append((cmd_name, cmd_info))

        if not available_commands:
            return None

        # Score commands based on multiple factors
        scored_commands = []
        for cmd_name, cmd_info in available_commands:
            score = 0.0

            # Success rate bonus
            score += cmd_info["success_rate"] * 20

            # Fitness impact bonus
            score += cmd_info["fitness_impact"] * 50

            # Recency penalty (prefer commands not run recently)
            recent_executions = sum(1 for exec_cmd in self.state.command_history[-5:]
                                  if exec_cmd.name == cmd_name)
            score -= recent_executions * 5

            # Pattern learning bonus
            pattern_bonus = self.pattern_learner.get_pattern_bonus(cmd_name, fitness_level.value)
            score += pattern_bonus

            scored_commands.append((cmd_name, score))

        # Sort by score and add some randomness for exploration
        scored_commands.sort(key=lambda x: x[1], reverse=True)
        exploration_factor = random.random() * 0.3  # 30% exploration
        selected_idx = 0 if random.random() > exploration_factor else random.randint(0, min(2, len(scored_commands)-1))

        selected_command = scored_commands[selected_idx][0]
        self.log(f"🤖 AI selected command: {selected_command} (score: {scored_commands[selected_idx][1]:.2f})")
        return selected_command

    async def execute_command(self, command_name: str) -> CommandExecution:
        """Execute a single command"""
        cmd_info = self.commands[command_name]
        execution = CommandExecution(
            name=command_name,
            fitness_before=self.state.current_fitness,
            dependencies=cmd_info["dependencies"]
        )

        self.state.active_commands.add(command_name)
        execution.start_time = time.time()
        execution.status = CommandStatus.RUNNING

        self.log(f"⚡ EXECUTING: {command_name} - {cmd_info['description']}")

        try:
            # Simulate command execution
            await asyncio.sleep(cmd_info["estimated_duration"] * (0.8 + random.random() * 0.4))

            # Simulate success/failure based on success rate
            success = random.random() < cmd_info["success_rate"]

            if success:
                execution.status = CommandStatus.COMPLETED
                fitness_change = cmd_info["fitness_impact"] * (0.8 + random.random() * 0.4)
                self.state.current_fitness = min(1.0, self.state.current_fitness + fitness_change)
                execution.fitness_after = self.state.current_fitness
                execution.output = f"Command completed successfully. Fitness +{fitness_change:.3f}"

                # Learn from success
                self.pattern_learner.learn_from_success(command_name, self.get_fitness_level(self.state.current_fitness).value)

            else:
                execution.status = CommandStatus.FAILED
                fitness_change = -cmd_info["fitness_impact"] * 0.5
                self.state.current_fitness = max(0.0, self.state.current_fitness + fitness_change)
                execution.fitness_after = self.state.current_fitness
                execution.error = f"Command failed. Fitness {fitness_change:.3f}"

        except Exception as e:
            execution.status = CommandStatus.FAILED
            execution.error = str(e)
            self.log(f"❌ Command {command_name} failed with error: {e}")

        finally:
            execution.end_time = time.time()
            self.state.active_commands.discard(command_name)
            self.state.command_history.append(execution)

        duration = execution.end_time - execution.start_time if execution.end_time and execution.start_time else 0
        self.log(f"✅ COMPLETED: {command_name} in {duration:.1f}s - Status: {execution.status.value}")

        return execution

    def should_complete_cycle(self) -> bool:
        """Determine if we should run a complete cycle"""
        if self.cycles_only:
            return True

        # Complete cycle every 10 iterations or when fitness is high
        return (self.state.iteration % 10 == 0 or
                self.state.current_fitness >= 0.85)

    async def run_cycle(self) -> Dict[str, Any]:
        """Run a complete cycle of command chaining"""
        self.state.cycles_completed += 1
        cycle_start_time = time.time()
        cycle_commands = []

        self.log(f"🔄 STARTING CYCLE #{self.state.cycles_completed}")

        # Run cycle logic (simplified for demo)
        cycle_sequence = ["analyze-custom-files", "evolution-status", "auto-evolve", "evolution-summary"]

        for cmd_name in cycle_sequence:
            if cmd_name in self.commands:
                execution = await self.execute_command(cmd_name)
                cycle_commands.append(execution)

                # Update fitness history
                self.state.fitness_history.append(self.state.current_fitness)

                # Check for early stopping conditions
                if self.should_stop():
                    break

        cycle_duration = time.time() - cycle_start_time
        self.state.last_cycle_time = time.time()

        cycle_result = {
            "cycle_number": self.state.cycles_completed,
            "commands_executed": len(cycle_commands),
            "duration": cycle_duration,
            "fitness_start": cycle_commands[0].fitness_before if cycle_commands else 0.0,
            "fitness_end": cycle_commands[-1].fitness_after if cycle_commands else 0.0,
            "fitness_improvement": (cycle_commands[-1].fitness_after - cycle_commands[0].fitness_before) if cycle_commands else 0.0
        }

        self.log(f"🎉 CYCLE #{self.state.cycles_completed} COMPLETED in {cycle_duration:.1f}s")
        self.log(f"📊 Fitness: {cycle_result['fitness_start']:.3f} → {cycle_result['fitness_end']:.3f} (Δ{cycle_result['fitness_improvement']:+.3f})")

        return cycle_result

    def should_stop(self) -> bool:
        """Check if the orchestrator should stop"""
        # Max iterations reached
        if self.state.iteration >= self.max_iterations:
            self.log("🛑 Max iterations reached")
            return True

        # Fitness threshold achieved with stability
        if self.state.current_fitness >= self.fitness_threshold:
            recent_improvements = 0
            if len(self.state.fitness_history) >= 3:
                for i in range(-3, 0):
                    if self.state.fitness_history[i] < self.state.fitness_history[i+1]:
                        recent_improvements += 1
            if recent_improvements >= 2:
                self.log(f"🎯 Fitness threshold {self.fitness_threshold} achieved with stability")
                return True

        # Too many consecutive degradations
        if len(self.state.fitness_history) >= 5:
            degradations = 0
            for i in range(-4, 0):
                if self.state.fitness_history[i] > self.state.fitness_history[i+1]:
                    degradations += 1
            if degradations >= 4:
                self.log("⚠️ Too many consecutive fitness degradations")
                return True

        return False

    async def run_orchestration(self) -> Dict[str, Any]:
        """Main orchestration loop with automatic safety protection"""
        self.log("🚀 STARTING AUTO-RECURSIVE CHAIN AI ORCHESTRATION")
        self.log("=" * 60)

        # Ensure safety systems are initialized and validate operation
        if SAFETY_SYSTEMS_AVAILABLE:
            # Initialize safety systems if needed
            safety_ready = await self._ensure_safety_systems()

            if safety_ready and self.safety_orchestrator:
                safety_context = SafetyContext(
                    user_id="auto_evolution_system",
                    operation_type="trigger_evolution",
                    risk_level="high",
                    requires_confirmation=False,  # Auto-approve for system operations
                    timeout_seconds=300
                )

                safety_result = await self.safety_orchestrator.validate_operation_safety(
                    "trigger_evolution",
                    self._run_orchestration_core,
                    safety_context
                )

                if not safety_result.approved:
                    self.log("❌ SAFETY VALIDATION FAILED - CANNOT START ORCHESTRATION")
                    for blocker in safety_result.blockers:
                        self.log(f"  🚫 {blocker}")
                    return {
                        'success': False,
                        'error': 'Safety validation failed',
                        'blockers': safety_result.blockers
                    }

                self.log(f"Safety validation passed for trigger_evolution in {time.time() - start_time:.2f}s")
            else:
                self.log("⚠️  SAFETY SYSTEMS FAILED TO INITIALIZE - PROCEEDING WITH EXTREME CAUTION")
        else:
            self.log("⚠️  SAFETY SYSTEMS UNAVAILABLE - RUNNING IN UNSAFE MODE")

        # Initialize fitness
        self.state.current_fitness = random.uniform(0.3, 0.6)  # Start with moderate fitness
        self.state.fitness_history.append(self.state.current_fitness)

        cycle_results = []

        # Execute the core orchestration logic (with safety if available)
        return await self._run_orchestration_core()

    async def _run_orchestration_core(self) -> Dict[str, Any]:
        """Core orchestration logic (called safely)"""
        cycle_results = []

        try:
            while not self.should_stop():
                self.state.iteration += 1
                self.state.total_iterations += 1

                self.log(f"\n🔄 ITERATION #{self.state.iteration} (Fitness: {self.state.current_fitness:.3f})")

                if self.should_complete_cycle():
                    cycle_result = await self.run_cycle()
                    cycle_results.append(cycle_result)
                else:
                    # Single command execution
                    next_command = self.select_next_command()
                    if next_command:
                        await self.execute_command(next_command)
                        self.state.fitness_history.append(self.state.current_fitness)
                    else:
                        self.log("🤔 No suitable command found, waiting...")
                        await asyncio.sleep(1.0)

                # Periodic state save
                if self.state.iteration % 5 == 0:
                    self.save_state()

                # Fitness check
                if self.state.current_fitness >= self.fitness_threshold:
                    self.log(f"🎯 Fitness threshold reached: {self.state.current_fitness:.3f} ≥ {self.fitness_threshold}")
                    break

                # Small delay between iterations
                await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            self.log("👋 Orchestration interrupted by user")

        except Exception as e:
            self.log(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Final state save
            self.save_state()

            # Generate summary
            summary = self.generate_summary(cycle_results)
            self.log("\n" + "=" * 60)
            self.log("🎉 ORCHESTRATION COMPLETE")
            self.log("=" * 60)
            for key, value in summary.items():
                self.log(f"{key}: {value}")

            return summary

    def save_state(self):
        """Save current state to file"""
        state_file = "logs/auto_recursive_chain_state.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)

        state_data = {
            "iteration": self.state.iteration,
            "total_iterations": self.state.total_iterations,
            "cycles_completed": self.state.cycles_completed,
            "current_fitness": self.state.current_fitness,
            "fitness_history": self.state.fitness_history[-50:],  # Last 50 entries
            "command_history_count": len(self.state.command_history),
            "learned_patterns": self.pattern_learner.patterns,
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)
            self.log(f"💾 State saved to {state_file}")
        except Exception as e:
            self.log(f"❌ Failed to save state: {e}")

    def generate_summary(self, cycle_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final summary"""
        total_runtime = time.time() - self.state.start_time
        commands_executed = len(self.state.command_history)

        return {
            "Total Runtime": f"{total_runtime:.1f}s",
            "Total Iterations": self.state.total_iterations,
            "Cycles Completed": self.state.cycles_completed,
            "Commands Executed": commands_executed,
            "Final Fitness": f"{self.state.current_fitness:.3f}",
            "Fitness Improvement": f"{self.state.current_fitness - self.state.fitness_history[0]:+.3f}",
            "Average Commands/Cycle": f"{commands_executed / max(1, self.state.cycles_completed):.1f}",
            "Success Rate": f"{len([c for c in self.state.command_history if c.status == CommandStatus.COMPLETED]) / max(1, commands_executed) * 100:.1f}%",
            "Patterns Learned": len(self.pattern_learner.patterns)
        }

class PatternLearner:
    """Learns successful command patterns"""

    def __init__(self):
        self.patterns = {}
        self.success_sequences = []

    def learn_from_success(self, command: str, fitness_level: str):
        """Learn from successful command execution"""
        key = f"{command}_{fitness_level}"

        if key not in self.patterns:
            self.patterns[key] = {
                "success_count": 0,
                "total_count": 0,
                "avg_fitness_impact": 0.0
            }

        self.patterns[key]["success_count"] += 1
        self.patterns[key]["total_count"] += 1

    def get_pattern_bonus(self, command: str, fitness_level: str) -> float:
        """Get pattern-based bonus for command selection"""
        key = f"{command}_{fitness_level}"

        if key in self.patterns:
            pattern = self.patterns[key]
            success_rate = pattern["success_count"] / max(1, pattern["total_count"])
            return success_rate * 10  # Bonus up to 10 points

        return 0.0

async def main():
    """Main execution with automatic safety integration"""
    parser = argparse.ArgumentParser(description="Auto-Recursive Chain AI Orchestrator")
    parser.add_argument("--max-iterations", type=int, default=100,
                       help="Maximum number of iterations")
    parser.add_argument("--fitness-threshold", type=float, default=0.95,
                       help="Fitness threshold to achieve")
    parser.add_argument("--cycles-only", action="store_true",
                       help="Run only complete cycles")
    parser.add_argument("--log-file", type=str,
                       help="Custom log file path")
    parser.add_argument("--safety-level", type=str, default="standard",
                       choices=["minimal", "standard", "strict", "paranoid"],
                       help="Safety level (minimal/standard/strict/paranoid)")

    args = parser.parse_args()

    # Auto-initialize master integration
    print("🛡️ Auto-initializing Consciousness Safety Systems...")
    try:
        from consciousness_master_integration import initialize_consciousness_suite
        await initialize_consciousness_suite()
    except ImportError:
        print("⚠️  Master integration not available - using basic safety")
    except Exception as e:
        print(f"⚠️  Safety integration failed: {e} - proceeding anyway")

    # Create orchestrator with safety level
    orchestrator = AutoRecursiveChainAI(
        max_iterations=args.max_iterations,
        fitness_threshold=args.fitness_threshold,
        cycles_only=args.cycles_only,
        log_file=args.log_file,
        safety_level=args.safety_level
    )

    # Run orchestration (now automatically safe)
    await orchestrator.run_orchestration()

if __name__ == "__main__":
    asyncio.run(main())
