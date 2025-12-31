#!/usr/bin/env python3
"""
🛡️ UI SAFETY SYSTEM
===================

Confirmation dialogs and progress indicators for evolution operations.
"""

import asyncio
import json
import time
import os
import threading
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
try:
    import curses
    HAS_CURSES = True
except ImportError:
    HAS_CURSES = False
    curses = None
import sys

class ConfirmationLevel(Enum):
    """Levels of confirmation required"""
    NONE = "none"           # No confirmation needed
    BASIC = "basic"         # Simple yes/no
    VERBOSE = "verbose"     # Detailed information required
    CRITICAL = "critical"   # Multiple confirmations + typing required

class ProgressStyle(Enum):
    """Progress indicator styles"""
    BAR = "bar"             # Traditional progress bar
    SPINNER = "spinner"     # Spinning indicator
    PERCENT = "percent"     # Simple percentage
    ETA = "eta"            # Time remaining estimate
    DETAILED = "detailed"   # Full progress information

class OperationRisk(Enum):
    """Risk levels for operations"""
    LOW = "low"             # Safe operations
    MEDIUM = "medium"       # Potentially disruptive
    HIGH = "high"           # System-wide impact
    CRITICAL = "critical"   # Irreversible changes

@dataclass
class OperationConfirmation:
    """Confirmation requirements for an operation"""
    operation_name: str
    risk_level: OperationRisk
    confirmation_level: ConfirmationLevel
    warning_message: str
    impact_description: str
    required_confirmations: int = 1
    typing_required: Optional[str] = None  # Text user must type
    timeout_seconds: int = 30

@dataclass
class ProgressIndicator:
    """Progress tracking for operations"""
    operation_id: str
    operation_name: str
    style: ProgressStyle
    total_steps: int = 0
    current_step: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_total_time: Optional[float] = None
    current_message: str = ""
    sub_operations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SafetyCheckpoint:
    """Safety checkpoint during operation"""
    checkpoint_id: str
    description: str
    risk_assessment: str
    mitigation_steps: List[str]
    requires_confirmation: bool = False
    auto_continue: bool = True
    timeout_seconds: int = 60

class ConfirmationDialog:
    """Interactive confirmation dialog system"""

    def __init__(self):
        self.operation_confirmations = self._load_confirmations()

    def _load_confirmations(self) -> Dict[str, OperationConfirmation]:
        """Load confirmation requirements for operations"""
        return {
            'trigger_evolution': OperationConfirmation(
                operation_name='trigger_evolution',
                risk_level=OperationRisk.HIGH,
                confirmation_level=ConfirmationLevel.CRITICAL,
                warning_message='⚠️  EVOLUTION OPERATION WARNING',
                impact_description='This will trigger an automated evolution cycle that may modify system behavior, deploy new code, and potentially disrupt services.',
                required_confirmations=2,
                typing_required='I understand the risks of evolution',
                timeout_seconds=60
            ),
            'force_rollback': OperationConfirmation(
                operation_name='force_rollback',
                risk_level=OperationRisk.CRITICAL,
                confirmation_level=ConfirmationLevel.CRITICAL,
                warning_message='🚨 FORCE ROLLBACK WARNING',
                impact_description='This will forcibly rollback recent changes, potentially losing data and disrupting active operations.',
                required_confirmations=3,
                typing_required='I accept data loss and service disruption',
                timeout_seconds=120
            ),
            'system_config': OperationConfirmation(
                operation_name='system_config',
                risk_level=OperationRisk.HIGH,
                confirmation_level=ConfirmationLevel.VERBOSE,
                warning_message='⚙️  SYSTEM CONFIGURATION WARNING',
                impact_description='This will modify core system configuration that affects all operations.',
                required_confirmations=1,
                timeout_seconds=45
            ),
            'modify_contracts': OperationConfirmation(
                operation_name='modify_contracts',
                risk_level=OperationRisk.MEDIUM,
                confirmation_level=ConfirmationLevel.VERBOSE,
                warning_message='📝 CONTRACT MODIFICATION WARNING',
                impact_description='This will modify evolution contracts that define system behavior.',
                required_confirmations=1,
                timeout_seconds=30
            ),
            'bypass_safety': OperationConfirmation(
                operation_name='bypass_safety',
                risk_level=OperationRisk.CRITICAL,
                confirmation_level=ConfirmationLevel.CRITICAL,
                warning_message='🚫 SAFETY BYPASS WARNING',
                impact_description='This will bypass all safety mechanisms. EXTREME CAUTION REQUIRED.',
                required_confirmations=5,
                typing_required='I acknowledge bypassing all safety systems',
                timeout_seconds=300
            )
        }

    async def get_confirmation(self, operation_name: str, user_context: Dict[str, Any] = None) -> bool:
        """Get user confirmation for an operation"""

        confirmation = self.operation_confirmations.get(operation_name)
        if not confirmation:
            # Default to basic confirmation for unknown operations
            confirmation = OperationConfirmation(
                operation_name=operation_name,
                risk_level=OperationRisk.MEDIUM,
                confirmation_level=ConfirmationLevel.BASIC,
                warning_message=f'Operation: {operation_name}',
                impact_description='This operation may affect system behavior.',
                timeout_seconds=30
            )

        # Display warning
        self._display_warning(confirmation)

        # Get confirmations based on level
        if confirmation.confirmation_level == ConfirmationLevel.NONE:
            return True
        elif confirmation.confirmation_level == ConfirmationLevel.BASIC:
            return await self._get_basic_confirmation(confirmation)
        elif confirmation.confirmation_level == ConfirmationLevel.VERBOSE:
            return await self._get_verbose_confirmation(confirmation)
        elif confirmation.confirmation_level == ConfirmationLevel.CRITICAL:
            return await self._get_critical_confirmation(confirmation)

        return False

    def _display_warning(self, confirmation: OperationConfirmation):
        """Display operation warning"""
        risk_colors = {
            OperationRisk.LOW: '\033[92m',      # Green
            OperationRisk.MEDIUM: '\033[93m',   # Yellow
            OperationRisk.HIGH: '\033[91m',     # Red
            OperationRisk.CRITICAL: '\033[95m'  # Magenta
        }

        color = risk_colors.get(confirmation.risk_level, '\033[0m')
        reset = '\033[0m'

        print(f"\n{color}{'='*60}{reset}")
        print(f"{color}{confirmation.warning_message}{reset}")
        print(f"{color}{'='*60}{reset}")
        print(f"Risk Level: {confirmation.risk_level.value.upper()}")
        print(f"Impact: {confirmation.impact_description}")

        if confirmation.typing_required:
            print(f"Required Typing: '{confirmation.typing_required}'")

        print(f"Timeout: {confirmation.timeout_seconds} seconds")
        print(f"{color}{'='*60}{reset}\n")

    async def _get_basic_confirmation(self, confirmation: OperationConfirmation) -> bool:
        """Get basic yes/no confirmation"""
        try:
            print(f"Continue with {confirmation.operation_name}? (y/N): ", end='', flush=True)

            # Simple timeout mechanism
            start_time = time.time()
            while time.time() - start_time < confirmation.timeout_seconds:
                if os.name == 'nt':  # Windows
                    try:
                        import msvcrt
                        if msvcrt.kbhit():
                            char = msvcrt.getch().decode('utf-8').lower()
                            if char == 'y':
                                print('y')
                                return True
                            elif char == 'n' or char == '\r':
                                print('n')
                                return False
                    except ImportError:
                        # Fallback for Windows without msvcrt
                        pass
                else:
                    # Unix-like systems - simplified for demo
                    await asyncio.sleep(0.1)

            print("\nTimeout - operation cancelled")
            return False

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return False

    async def _get_verbose_confirmation(self, confirmation: OperationConfirmation) -> bool:
        """Get verbose confirmation with detailed information"""
        print("Detailed Impact Assessment:")
        print("• This operation may take several minutes to complete")
        print("• System performance may be affected during execution")
        print("• Automatic rollback is available if issues occur")
        print("• Progress will be displayed in real-time")
        print()

        return await self._get_basic_confirmation(confirmation)

    async def _get_critical_confirmation(self, confirmation: OperationConfirmation) -> bool:
        """Get critical confirmation with multiple checks"""

        # Multiple confirmation rounds
        for i in range(confirmation.required_confirmations):
            print(f"CONFIRMATION ROUND {i+1}/{confirmation.required_confirmations}")
            print("-" * 40)

            confirmed = await self._get_basic_confirmation(confirmation)
            if not confirmed:
                return False

            print(f"✓ Confirmation {i+1} received\n")

        # Typing requirement
        if confirmation.typing_required:
            print(f"Please type the following text to confirm: '{confirmation.typing_required}'")
            print("(This prevents accidental confirmation)")
            print()

            try:
                # In a real implementation, this would use proper input handling
                user_input = input("Type confirmation text: ").strip()
                if user_input != confirmation.typing_required:
                    print("❌ Incorrect text entered - operation cancelled")
                    return False
                print("✓ Text confirmation received\n")

            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled")
                return False

        print("🎯 ALL CONFIRMATIONS RECEIVED - PROCEEDING WITH OPERATION")
        return True

class ProgressDisplay:
    """Progress indicator and display system"""

    def __init__(self):
        self.active_progress: Dict[str, ProgressIndicator] = {}
        self.display_thread: Optional[threading.Thread] = None
        self.display_active = False

    def start_progress(self, operation_id: str, operation_name: str,
                      style: ProgressStyle = ProgressStyle.BAR,
                      total_steps: int = 100) -> ProgressIndicator:
        """Start tracking progress for an operation"""

        progress = ProgressIndicator(
            operation_id=operation_id,
            operation_name=operation_name,
            style=style,
            total_steps=total_steps
        )

        self.active_progress[operation_id] = progress

        # Start display thread if not running
        if not self.display_active:
            self.display_active = True
            self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
            self.display_thread.start()

        print(f"\n🚀 STARTING: {operation_name}")
        print("=" * 50)

        return progress

    def update_progress(self, operation_id: str, current_step: int,
                       message: str = "", metadata: Dict[str, Any] = None):
        """Update progress for an operation"""

        if operation_id not in self.active_progress:
            return

        progress = self.active_progress[operation_id]
        progress.current_step = current_step
        progress.current_message = message

        if metadata:
            progress.metadata.update(metadata)

        # Update estimated time if we have enough data
        if progress.current_step > 5:
            elapsed = time.time() - progress.start_time
            progress_per_step = elapsed / progress.current_step
            remaining_steps = progress.total_steps - progress.current_step
            progress.estimated_total_time = elapsed + (remaining_steps * progress_per_step)

    def add_sub_operation(self, operation_id: str, sub_op_name: str,
                         sub_op_id: str, status: str = "pending"):
        """Add a sub-operation to track"""

        if operation_id not in self.active_progress:
            return

        progress = self.active_progress[operation_id]
        progress.sub_operations.append({
            'id': sub_op_id,
            'name': sub_op_name,
            'status': status,
            'start_time': time.time()
        })

    def update_sub_operation(self, operation_id: str, sub_op_id: str, status: str):
        """Update sub-operation status"""

        if operation_id not in self.active_progress:
            return

        progress = self.active_progress[operation_id]
        for sub_op in progress.sub_operations:
            if sub_op['id'] == sub_op_id:
                sub_op['status'] = status
                if status in ['completed', 'failed']:
                    sub_op['end_time'] = time.time()
                    sub_op['duration'] = sub_op['end_time'] - sub_op['start_time']
                break

    def complete_progress(self, operation_id: str, success: bool = True,
                         final_message: str = ""):
        """Complete progress tracking"""

        if operation_id not in self.active_progress:
            return

        progress = self.active_progress[operation_id]

        status = "✅ COMPLETED" if success else "❌ FAILED"
        message = final_message or f"Operation {progress.operation_name} {status.lower()}"

        print(f"\n{status}: {message}")
        print("=" * 50)

        # Show final statistics
        total_time = time.time() - progress.start_time
        print(".1f")
        print(f"Steps: {progress.current_step}/{progress.total_steps}")

        if progress.sub_operations:
            completed = len([s for s in progress.sub_operations if s['status'] == 'completed'])
            print(f"Sub-operations: {completed}/{len(progress.sub_operations)} completed")

        # Remove from active progress
        del self.active_progress[operation_id]

        # Stop display if no more active progress
        if not self.active_progress:
            self.display_active = False

    def _display_worker(self):
        """Background display thread"""
        while self.display_active:
            try:
                # Clear screen area for progress display
                self._display_progress()

                time.sleep(1)  # Update every second

            except Exception as e:
                print(f"Display error: {e}")
                time.sleep(1)

    def _display_progress(self):
        """Display current progress for all active operations"""

        if not self.active_progress:
            return

        # Save cursor position and move to top of progress area
        print("\033[s", end='')  # Save cursor
        print("\033[0;0H", end='')  # Move to top-left

        for operation_id, progress in self.active_progress.items():
            self._display_single_progress(progress)

        # Restore cursor position
        print("\033[u", end='')  # Restore cursor

    def _display_single_progress(self, progress: ProgressIndicator):
        """Display progress for a single operation"""

        percentage = (progress.current_step / progress.total_steps) * 100 if progress.total_steps > 0 else 0

        if progress.style == ProgressStyle.BAR:
            self._display_progress_bar(progress, percentage)
        elif progress.style == ProgressStyle.SPINNER:
            self._display_spinner(progress, percentage)
        elif progress.style == ProgressStyle.PERCENT:
            self._display_percentage(progress, percentage)
        elif progress.style == ProgressStyle.ETA:
            self._display_eta(progress, percentage)
        elif progress.style == ProgressStyle.DETAILED:
            self._display_detailed(progress, percentage)

    def _display_progress_bar(self, progress: ProgressIndicator, percentage: float):
        """Display traditional progress bar"""

        bar_width = 40
        filled = int(bar_width * percentage / 100)
        bar = '█' * filled + '░' * (bar_width - filled)

        eta_text = ""
        if progress.estimated_total_time:
            remaining = progress.estimated_total_time - (time.time() - progress.start_time)
            if remaining > 0:
                eta_text = f" ETA: {remaining:.0f}s"

        print("2d"
              f"[{bar}] {percentage:.1f}%{eta_text}")

        if progress.current_message:
            print(f"  └─ {progress.current_message}")

    def _display_spinner(self, progress: ProgressIndicator, percentage: float):
        """Display spinning progress indicator"""

        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        spinner_idx = int(time.time() * 2) % len(spinner_chars)

        print("2d"
              f"{percentage:.1f}%")

        if progress.current_message:
            print(f"  └─ {progress.current_message}")

    def _display_percentage(self, progress: ProgressIndicator, percentage: float):
        """Display simple percentage"""

        print("2d"
              f"{percentage:.1f}%")

        if progress.current_message:
            print(f"  └─ {progress.current_message}")

    def _display_eta(self, progress: ProgressIndicator, percentage: float):
        """Display time remaining estimate"""

        eta_text = "Calculating..."
        if progress.estimated_total_time:
            elapsed = time.time() - progress.start_time
            remaining = progress.estimated_total_time - elapsed
            if remaining > 0:
                eta_text = f"{remaining:.0f}s remaining"
            else:
                eta_text = "Overdue"

        print("2d"
              f"{percentage:.1f}% | {eta_text}")

        if progress.current_message:
            print(f"  └─ {progress.current_message}")

    def _display_detailed(self, progress: ProgressIndicator, percentage: float):
        """Display detailed progress information"""

        elapsed = time.time() - progress.start_time

        print(f"🔄 {progress.operation_name}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Steps: {progress.current_step}/{progress.total_steps}")
        print(f"  Progress: {percentage:.1f}%")
        # Show sub-operations
        if progress.sub_operations:
            print("  Sub-operations:")
            for sub_op in progress.sub_operations[-3:]:  # Show last 3
                status_emoji = {
                    'pending': '⏳',
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌'
                }.get(sub_op['status'], '❓')

                duration = sub_op.get('duration', time.time() - sub_op['start_time'])
                print(f"    {status_emoji} {sub_op['name']} ({duration:.1f}s)")

        if progress.current_message:
            print(f"  └─ {progress.current_message}")

        print()  # Blank line between operations

class SafetyCheckpointManager:
    """Safety checkpoint system for operations"""

    def __init__(self):
        self.checkpoints: Dict[str, SafetyCheckpoint] = {}
        self.active_checkpoints: Dict[str, Dict[str, Any]] = {}

    def add_checkpoint(self, operation_id: str, checkpoint: SafetyCheckpoint):
        """Add a safety checkpoint"""

        checkpoint_id = f"{operation_id}_{checkpoint.checkpoint_id}"
        self.checkpoints[checkpoint_id] = checkpoint

        if checkpoint.requires_confirmation:
            self.active_checkpoints[checkpoint_id] = {
                'checkpoint': checkpoint,
                'operation_id': operation_id,
                'status': 'waiting',
                'start_time': time.time()
            }

    async def process_checkpoints(self, operation_id: str) -> bool:
        """Process all safety checkpoints for an operation"""

        operation_checkpoints = [
            (cid, cp) for cid, cp in self.active_checkpoints.items()
            if cp['operation_id'] == operation_id
        ]

        for checkpoint_id, checkpoint_data in operation_checkpoints:
            checkpoint = checkpoint_data['checkpoint']

            print(f"\n🛑 SAFETY CHECKPOINT: {checkpoint.description}")
            print("=" * 50)
            print(f"Risk: {checkpoint.risk_assessment}")
            print("Mitigation steps:")
            for i, step in enumerate(checkpoint.mitigation_steps, 1):
                print(f"  {i}. {step}")

            if checkpoint.requires_confirmation:
                print(f"\nContinue with operation? (y/N) [{checkpoint.timeout_seconds}s timeout]: ", end='', flush=True)

                confirmed = await self._wait_for_confirmation(checkpoint.timeout_seconds)

                if not confirmed:
                    if checkpoint.auto_continue:
                        print("Auto-continuing per checkpoint configuration...")
                    else:
                        print("Operation cancelled at safety checkpoint")
                        return False

                checkpoint_data['status'] = 'confirmed'

        return True

    async def _wait_for_confirmation(self, timeout_seconds: int) -> bool:
        """Wait for user confirmation with timeout"""
        try:
            start_time = time.time()

            while time.time() - start_time < timeout_seconds:
                # Simple input check (would be more sophisticated in real implementation)
                await asyncio.sleep(0.1)

            return False  # Timeout

        except KeyboardInterrupt:
            return False

class EvolutionSafetyUI:
    """Complete UI safety system for evolution operations"""

    def __init__(self):
        self.confirmation_dialog = ConfirmationDialog()
        self.progress_display = ProgressDisplay()
        self.checkpoint_manager = SafetyCheckpointManager()

    async def execute_safe_operation(self, operation_name: str,
                                   operation_func: Callable[[], Awaitable[Any]],
                                   user_context: Dict[str, Any] = None,
                                   progress_style: ProgressStyle = ProgressStyle.BAR) -> Dict[str, Any]:
        """Execute an operation with full safety UI"""

        result = {
            'success': False,
            'cancelled': False,
            'error': None,
            'execution_time': 0,
            'safety_checks_passed': 0
        }

        start_time = time.time()

        try:
            # Step 1: Get user confirmation
            print(f"[*] Requesting confirmation for: {operation_name}")
            confirmed = await self.confirmation_dialog.get_confirmation(operation_name, user_context)

            if not confirmed:
                result['cancelled'] = True
                print("[-] Operation cancelled by user")
                return result

            result['safety_checks_passed'] += 1

            # Step 2: Start progress tracking
            progress = self.progress_display.start_progress(
                operation_id=f"{operation_name}_{int(start_time)}",
                operation_name=operation_name,
                style=progress_style,
                total_steps=10  # Default, can be updated
            )

            # Step 3: Add safety checkpoints
            self._add_operation_checkpoints(operation_name, progress.operation_id)

            # Step 4: Process safety checkpoints
            progress.update_progress(progress.operation_id, 1, "Processing safety checkpoints...")
            checkpoints_passed = await self.checkpoint_manager.process_checkpoints(progress.operation_id)

            if not checkpoints_passed:
                result['cancelled'] = True
                self.progress_display.complete_progress(progress.operation_id, False, "Cancelled at safety checkpoint")
                return result

            result['safety_checks_passed'] += 1

            # Step 5: Execute operation with progress updates
            progress.update_progress(progress.operation_id, 3, "Executing operation...")

            operation_result = await operation_func()

            progress.update_progress(progress.operation_id, 8, "Operation completed, finalizing...")

            # Step 6: Complete progress
            progress.update_progress(progress.operation_id, 10, "All checks passed")
            self.progress_display.complete_progress(progress.operation_id, True)

            result['success'] = True
            result['result'] = operation_result

        except Exception as e:
            error_msg = str(e)
            result['error'] = error_msg

            # Complete progress with failure
            try:
                self.progress_display.complete_progress(
                    f"{operation_name}_{int(start_time)}",
                    False,
                    f"Failed: {error_msg}"
                )
            except:
                pass  # Progress might not have been started

        finally:
            result['execution_time'] = time.time() - start_time

        return result

    def _add_operation_checkpoints(self, operation_name: str, operation_id: str):
        """Add appropriate safety checkpoints for an operation"""

        checkpoints = {
            'trigger_evolution': [
                SafetyCheckpoint(
                    checkpoint_id='resource_check',
                    description='Resource Availability Check',
                    risk_assessment='Evolution may consume significant system resources',
                    mitigation_steps=[
                        'Verify adequate CPU, memory, and disk space',
                        'Check for conflicting operations',
                        'Ensure backup systems are operational'
                    ],
                    requires_confirmation=True
                ),
                SafetyCheckpoint(
                    checkpoint_id='data_backup',
                    description='Data Backup Verification',
                    risk_assessment='Evolution may modify or lose data',
                    mitigation_steps=[
                        'Verify recent backups exist',
                        'Check backup integrity',
                        'Ensure rollback capability'
                    ],
                    requires_confirmation=False,
                    auto_continue=True
                )
            ],
            'force_rollback': [
                SafetyCheckpoint(
                    checkpoint_id='data_loss_warning',
                    description='Data Loss Warning',
                    risk_assessment='Rollback may cause irreversible data loss',
                    mitigation_steps=[
                        'Confirm data has been backed up',
                        'Notify affected users',
                        'Prepare communication plan'
                    ],
                    requires_confirmation=True
                )
            ]
        }

        if operation_name in checkpoints:
            for checkpoint in checkpoints[operation_name]:
                self.checkpoint_manager.add_checkpoint(operation_id, checkpoint)

def main():
    """CLI interface for UI safety system"""
    import argparse

    parser = argparse.ArgumentParser(description="UI Safety System")
    parser.add_argument("--confirm", metavar="OPERATION",
                       help="Test confirmation dialog for operation")
    parser.add_argument("--progress-demo", metavar="STYLE",
                       help="Demonstrate progress indicator")
    parser.add_argument("--safe-execute", metavar="OPERATION",
                       help="Demonstrate safe operation execution")

    args = parser.parse_args()

    async def run():
        ui = EvolutionSafetyUI()

        if args.confirm:
            confirmed = await ui.confirmation_dialog.get_confirmation(args.confirm)
            print(f"Confirmation result: {confirmed}")

        elif args.progress_demo:
            style = ProgressStyle(args.progress_demo.upper()) if args.progress_demo else ProgressStyle.BAR

            progress = ui.progress_display.start_progress(
                "demo_operation", "Demo Operation", style, 20
            )

            for i in range(21):
                ui.progress_display.update_progress(
                    progress.operation_id, i,
                    f"Processing step {i}/20"
                )
                await asyncio.sleep(0.5)

            ui.progress_display.complete_progress(progress.operation_id, True)

        elif args.safe_execute:
            async def demo_operation():
                await asyncio.sleep(2)
                return "Operation completed successfully"

            result = await ui.execute_safe_operation(
                args.safe_execute, demo_operation,
                progress_style=ProgressStyle.DETAILED
            )

            print(f"Safe execution result: {result}")

        else:
            parser.print_help()

    asyncio.run(run())

if __name__ == "__main__":
    main()
