#!/usr/bin/env python3
"""
🔄 TRANSACTIONAL EVOLUTION SYSTEM
=================================

Atomic evolution operations with automatic rollback capabilities.
"""

import asyncio
import json
import time
import os
import shutil
import hashlib
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import logging

class TransactionStatus(Enum):
    """Status of evolution transaction"""
    PENDING = "pending"
    RUNNING = "running"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

class TransactionPhase(Enum):
    """Phases of evolution transaction"""
    PREPARE = "prepare"
    VALIDATE = "validate"
    BACKUP = "backup"
    EXECUTE = "execute"
    VERIFY = "verify"
    COMMIT = "commit"
    CLEANUP = "cleanup"

@dataclass
class EvolutionStep:
    """Single step in evolution transaction"""
    name: str
    phase: TransactionPhase
    execute_func: Callable[[], Awaitable[Any]]
    rollback_func: Optional[Callable[[], Awaitable[Any]]] = None
    validate_func: Optional[Callable[[Any], Awaitable[bool]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    result: Any = None
    error: Optional[str] = None

@dataclass
class EvolutionTransaction:
    """Complete evolution transaction"""
    transaction_id: str
    description: str
    steps: List[EvolutionStep] = field(default_factory=list)
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_phase: TransactionPhase = TransactionPhase.PREPARE
    backup_path: Optional[str] = None
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransactionResult:
    """Result of transaction execution"""
    success: bool
    transaction: EvolutionTransaction
    error_message: Optional[str] = None
    rollback_performed: bool = False
    execution_time: float = 0.0

class EvolutionStateManager:
    """Manages evolution state snapshots for rollback"""

    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def create_backup(self, transaction_id: str, state_data: Dict[str, Any]) -> str:
        """Create backup of current evolution state"""
        backup_path = os.path.join(self.backup_dir, f"backup_{transaction_id}_{int(time.time())}")

        # Create backup directory
        os.makedirs(backup_path, exist_ok=True)

        # Backup key files and state
        backup_files = {
            'auto_recursive_chain_state.json': 'logs/auto_recursive_chain_state.json',
            'verified_evolution_state.json': 'verified_evolution_state.json',
            'consciousness_evolution_contract.json': 'consciousness_evolution_contract.json',
            'current_fitness.txt': None  # Will be created
        }

        for backup_name, source_path in backup_files.items():
            backup_file = os.path.join(backup_path, backup_name)

            if source_path and os.path.exists(source_path):
                shutil.copy2(source_path, backup_file)
            elif backup_name == 'current_fitness.txt':
                # Create fitness snapshot
                fitness = state_data.get('fitness', 0.0)
                with open(backup_file, 'w') as f:
                    f.write(str(fitness))

        # Create metadata
        metadata = {
            'transaction_id': transaction_id,
            'created_at': time.time(),
            'state_data': state_data,
            'backup_files': list(backup_files.keys())
        }

        with open(os.path.join(backup_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return backup_path

    def restore_backup(self, backup_path: str) -> bool:
        """Restore state from backup"""
        metadata_file = os.path.join(backup_path, 'metadata.json')

        if not os.path.exists(metadata_file):
            return False

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Restore files
            backup_files = metadata.get('backup_files', [])
            for filename in backup_files:
                backup_file = os.path.join(backup_path, filename)
                target_file = os.path.join('logs' if 'state' in filename else '', filename.replace('logs/', ''))

                if os.path.exists(backup_file):
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)
                    shutil.copy2(backup_file, target_file)

            print(f"[+] Backup restored from {backup_path}")
            return True

        except Exception as e:
            print(f"[-] Failed to restore backup: {e}")
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []

        if os.path.exists(self.backup_dir):
            for item in os.listdir(self.backup_dir):
                backup_path = os.path.join(self.backup_dir, item)
                if os.path.isdir(backup_path):
                    metadata_file = os.path.join(backup_path, 'metadata.json')
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            backups.append({
                                'path': backup_path,
                                'transaction_id': metadata.get('transaction_id'),
                                'created_at': metadata.get('created_at'),
                                'state_data': metadata.get('state_data', {})
                            })
                        except:
                            pass

        return sorted(backups, key=lambda x: x['created_at'], reverse=True)

class TransactionalEvolutionManager:
    """Manages transactional evolution operations"""

    def __init__(self, transaction_dir: str = "transactions"):
        self.transaction_dir = transaction_dir
        self.state_manager = EvolutionStateManager()
        self.active_transactions: Dict[str, EvolutionTransaction] = {}

        os.makedirs(transaction_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger('TransactionalEvolution')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler('logs/transactional_evolution.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def create_transaction(self, description: str, steps: List[EvolutionStep],
                          initial_state: Dict[str, Any]) -> EvolutionTransaction:
        """Create a new evolution transaction"""

        transaction_id = f"txn_{int(time.time())}_{hashlib.md5(description.encode()).hexdigest()[:8]}"

        transaction = EvolutionTransaction(
            transaction_id=transaction_id,
            description=description,
            steps=steps,
            fitness_before=initial_state.get('fitness', 0.0),
            metadata={
                'initial_state': initial_state,
                'created_by': 'TransactionalEvolutionManager'
            }
        )

        self.active_transactions[transaction_id] = transaction
        self._save_transaction(transaction)

        self.logger.info(f"Created transaction {transaction_id}: {description}")
        return transaction

    async def execute_transaction(self, transaction: EvolutionTransaction) -> TransactionResult:
        """Execute evolution transaction with rollback capabilities"""

        start_time = time.time()
        transaction.started_at = start_time
        transaction.status = TransactionStatus.RUNNING

        self.logger.info(f"Starting transaction {transaction.transaction_id}")

        try:
            # Phase 1: Prepare
            transaction.current_phase = TransactionPhase.PREPARE
            await self._prepare_transaction(transaction)

            # Phase 2: Validate
            transaction.current_phase = TransactionPhase.VALIDATE
            await self._validate_transaction(transaction)

            # Phase 3: Backup
            transaction.current_phase = TransactionPhase.BACKUP
            transaction.backup_path = self.state_manager.create_backup(
                transaction.transaction_id,
                transaction.metadata.get('initial_state', {})
            )

            # Phase 4: Execute steps
            transaction.current_phase = TransactionPhase.EXECUTE
            await self._execute_steps(transaction)

            # Phase 5: Verify
            transaction.current_phase = TransactionPhase.VERIFY
            await self._verify_transaction(transaction)

            # Phase 6: Commit
            transaction.current_phase = TransactionPhase.COMMIT
            await self._commit_transaction(transaction)

            # Phase 7: Cleanup
            transaction.current_phase = TransactionPhase.CLEANUP
            await self._cleanup_transaction(transaction)

            transaction.status = TransactionStatus.COMMITTED
            transaction.completed_at = time.time()
            transaction.fitness_after = await self._measure_fitness()

            execution_time = time.time() - start_time

            self.logger.info(f"Transaction {transaction.transaction_id} committed successfully")

            return TransactionResult(
                success=True,
                transaction=transaction,
                execution_time=execution_time
            )

        except Exception as e:
            self.logger.error(f"Transaction {transaction.transaction_id} failed: {e}")

            # Rollback transaction
            await self._rollback_transaction(transaction)

            transaction.status = TransactionStatus.FAILED
            transaction.completed_at = time.time()

            return TransactionResult(
                success=False,
                transaction=transaction,
                error_message=str(e),
                rollback_performed=True,
                execution_time=time.time() - start_time
            )

    async def _prepare_transaction(self, transaction: EvolutionTransaction):
        """Prepare transaction for execution"""
        self.logger.info(f"Preparing transaction {transaction.transaction_id}")

        # Validate all steps have required functions
        for step in transaction.steps:
            if not step.execute_func:
                raise ValueError(f"Step {step.name} missing execute function")

        # Check system health
        health_ok = await self._check_system_health()
        if not health_ok:
            raise RuntimeError("System health check failed")

    async def _validate_transaction(self, transaction: EvolutionTransaction):
        """Validate transaction before execution"""
        self.logger.info(f"Validating transaction {transaction.transaction_id}")

        # Validate step dependencies
        step_names = {step.name for step in transaction.steps}
        for step in transaction.steps:
            # Check if step can be validated
            pass  # Additional validation logic can be added here

    async def _execute_steps(self, transaction: EvolutionTransaction):
        """Execute all transaction steps"""
        self.logger.info(f"Executing steps for transaction {transaction.transaction_id}")

        for step in transaction.steps:
            self.logger.info(f"Executing step: {step.name}")

            try:
                # Execute step
                step.result = await step.execute_func()
                step.executed = True

                # Validate result if validator provided
                if step.validate_func:
                    is_valid = await step.validate_func(step.result)
                    if not is_valid:
                        raise ValueError(f"Step {step.name} validation failed")

                self.logger.info(f"Step {step.name} completed successfully")

            except Exception as e:
                step.error = str(e)
                raise RuntimeError(f"Step {step.name} failed: {e}")

    async def _verify_transaction(self, transaction: EvolutionTransaction):
        """Verify transaction results"""
        self.logger.info(f"Verifying transaction {transaction.transaction_id}")

        # Check that all steps completed successfully
        failed_steps = [step for step in transaction.steps if step.error]
        if failed_steps:
            raise RuntimeError(f"Verification failed: {len(failed_steps)} steps failed")

        # Verify system integrity
        integrity_ok = await self._verify_system_integrity()
        if not integrity_ok:
            raise RuntimeError("System integrity verification failed")

    async def _commit_transaction(self, transaction: EvolutionTransaction):
        """Commit transaction"""
        self.logger.info(f"Committing transaction {transaction.transaction_id}")

        transaction.status = TransactionStatus.COMMITTED
        self._save_transaction(transaction)

        # Clean up backup (transaction successful)
        if transaction.backup_path:
            # Keep backup for audit purposes but mark as committed
            pass

    async def _cleanup_transaction(self, transaction: EvolutionTransaction):
        """Clean up after successful transaction"""
        self.logger.info(f"Cleaning up transaction {transaction.transaction_id}")

        # Remove from active transactions
        if transaction.transaction_id in self.active_transactions:
            del self.active_transactions[transaction.transaction_id]

    async def _rollback_transaction(self, transaction: EvolutionTransaction):
        """Rollback failed transaction"""
        self.logger.warning(f"Rolling back transaction {transaction.transaction_id}")

        transaction.status = TransactionStatus.ROLLED_BACK

        try:
            # Restore from backup
            if transaction.backup_path:
                success = self.state_manager.restore_backup(transaction.backup_path)
                if success:
                    self.logger.info("Backup restored successfully")
                else:
                    self.logger.error("Backup restoration failed")

            # Rollback individual steps in reverse order
            for step in reversed(transaction.steps):
                if step.executed and step.rollback_func:
                    try:
                        await step.rollback_func()
                        self.logger.info(f"Rolled back step: {step.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to rollback step {step.name}: {e}")

            transaction.fitness_after = await self._measure_fitness()
            self._save_transaction(transaction)

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            transaction.metadata['rollback_error'] = str(e)

    async def _check_system_health(self) -> bool:
        """Check overall system health"""
        # Simulate health checks
        try:
            # Check if critical files exist
            critical_files = [
                'consciousness_evolution_contract.json',
                'verified_evolution_state.json'
            ]

            for file in critical_files:
                if not os.path.exists(file):
                    return False

            return True
        except:
            return False

    async def _verify_system_integrity(self) -> bool:
        """Verify system integrity after transaction"""
        # Simulate integrity checks
        try:
            # Check if evolution state is valid JSON
            if os.path.exists('logs/auto_recursive_chain_state.json'):
                with open('logs/auto_recursive_chain_state.json', 'r') as f:
                    json.load(f)

            return True
        except:
            return False

    async def _measure_fitness(self) -> float:
        """Measure current system fitness"""
        # Try to read from state file
        try:
            if os.path.exists('logs/auto_recursive_chain_state.json'):
                with open('logs/auto_recursive_chain_state.json', 'r') as f:
                    state = json.load(f)
                    return state.get('current_fitness', 0.0)
        except:
            pass

        return 0.0

    def _save_transaction(self, transaction: EvolutionTransaction):
        """Save transaction to disk"""
        transaction_file = os.path.join(self.transaction_dir, f"{transaction.transaction_id}.json")

        # Convert to serializable format
        transaction_data = {
            'transaction_id': transaction.transaction_id,
            'description': transaction.description,
            'status': transaction.status.value,
            'created_at': transaction.created_at,
            'started_at': transaction.started_at,
            'completed_at': transaction.completed_at,
            'current_phase': transaction.current_phase.value,
            'backup_path': transaction.backup_path,
            'fitness_before': transaction.fitness_before,
            'fitness_after': transaction.fitness_after,
            'metadata': transaction.metadata,
            'steps': [{
                'name': step.name,
                'phase': step.phase.value,
                'executed': step.executed,
                'error': step.error,
                'metadata': step.metadata
            } for step in transaction.steps]
        }

        with open(transaction_file, 'w') as f:
            json.dump(transaction_data, f, indent=2, default=str)

    def load_transaction(self, transaction_id: str) -> Optional[EvolutionTransaction]:
        """Load transaction from disk"""
        transaction_file = os.path.join(self.transaction_dir, f"{transaction_id}.json")

        if not os.path.exists(transaction_file):
            return None

        try:
            with open(transaction_file, 'r') as f:
                data = json.load(f)

            transaction = EvolutionTransaction(
                transaction_id=data['transaction_id'],
                description=data['description'],
                status=TransactionStatus(data['status']),
                created_at=data['created_at'],
                started_at=data.get('started_at'),
                completed_at=data.get('completed_at'),
                current_phase=TransactionPhase(data['current_phase']),
                backup_path=data.get('backup_path'),
                fitness_before=data['fitness_before'],
                fitness_after=data['fitness_after'],
                metadata=data.get('metadata', {})
            )

            # Note: steps are not fully restored as they contain functions
            return transaction

        except Exception as e:
            self.logger.error(f"Failed to load transaction {transaction_id}: {e}")
            return None

    def list_transactions(self) -> List[EvolutionTransaction]:
        """List all transactions"""
        transactions = []

        if os.path.exists(self.transaction_dir):
            for filename in os.listdir(self.transaction_dir):
                if filename.endswith('.json'):
                    transaction_id = filename[:-5]  # Remove .json
                    transaction = self.load_transaction(transaction_id)
                    if transaction:
                        transactions.append(transaction)

        return sorted(transactions, key=lambda t: t.created_at, reverse=True)

# Convenience functions for creating evolution steps
def create_evolution_step(name: str, execute_func: Callable[[], Awaitable[Any]],
                         rollback_func: Optional[Callable[[], Awaitable[Any]]] = None,
                         validate_func: Optional[Callable[[Any], Awaitable[bool]]] = None) -> EvolutionStep:
    """Create an evolution step"""
    return EvolutionStep(
        name=name,
        phase=TransactionPhase.EXECUTE,
        execute_func=execute_func,
        rollback_func=rollback_func,
        validate_func=validate_func
    )

async def demo_transactional_evolution():
    """Demonstrate transactional evolution system"""

    print("🔄 TRANSACTIONAL EVOLUTION DEMONSTRATION")
    print("=" * 50)

    manager = TransactionalEvolutionManager()

    # Create sample evolution steps
    async def step1():
        print("  Executing step 1: Update configuration")
        await asyncio.sleep(0.5)
        # Simulate configuration update
        return "config_updated"

    async def rollback_step1():
        print("  Rolling back step 1: Restore configuration")
        await asyncio.sleep(0.2)
        return "config_restored"

    async def validate_step1(result):
        return result == "config_updated"

    async def step2():
        print("  Executing step 2: Apply evolution algorithm")
        await asyncio.sleep(0.8)
        # Simulate evolution application
        if random.random() < 0.7:  # 70% success rate
            return "evolution_applied"
        else:
            raise Exception("Evolution algorithm failed")

    async def rollback_step2():
        print("  Rolling back step 2: Revert evolution changes")
        await asyncio.sleep(0.3)
        return "evolution_reverted"

    async def step3():
        print("  Executing step 3: Validate results")
        await asyncio.sleep(0.3)
        return "validation_passed"

    # Create steps
    steps = [
        create_evolution_step("update_config", step1, rollback_step1, validate_step1),
        create_evolution_step("apply_evolution", step2, rollback_step2),
        create_evolution_step("validate_results", step3)
    ]

    # Create transaction
    initial_state = {'fitness': 0.8, 'version': '1.0.0'}
    transaction = manager.create_transaction(
        "Demonstration evolution transaction",
        steps,
        initial_state
    )

    print(f"Created transaction: {transaction.transaction_id}")
    print(f"Steps: {len(transaction.steps)}")

    # Execute transaction
    print("\n🚀 EXECUTING TRANSACTION...")
    result = await manager.execute_transaction(transaction)

    print(f"\n📊 RESULT: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Execution time: {result.execution_time:.2f}s")

    if result.rollback_performed:
        print("🔙 Rollback was performed due to failure")

    if result.error_message:
        print(f"Error: {result.error_message}")

    print(f"Final fitness: {transaction.fitness_after:.3f}")

    # List transactions
    print(f"\n📋 Total transactions: {len(manager.list_transactions())}")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Transactional Evolution Manager")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--list", action="store_true", help="List all transactions")
    parser.add_argument("--backup-list", action="store_true", help="List all backups")

    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo_transactional_evolution())
    elif args.list:
        manager = TransactionalEvolutionManager()
        transactions = manager.list_transactions()

        if transactions:
            print("📋 TRANSACTIONS:")
            for txn in transactions[:5]:  # Show last 5
                status_emoji = {
                    TransactionStatus.COMMITTED: "✅",
                    TransactionStatus.ROLLED_BACK: "🔙",
                    TransactionStatus.FAILED: "❌",
                    TransactionStatus.RUNNING: "🔄"
                }.get(txn.status, "❓")

                print(f"{status_emoji} {txn.transaction_id} - {txn.description}")
                print(f"   Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(txn.created_at))}")
                if txn.completed_at:
                    duration = txn.completed_at - (txn.started_at or txn.created_at)
                    print(f"   Duration: {duration:.2f}s")
                print()
        else:
            print("No transactions found")

    elif args.backup_list:
        manager = TransactionalEvolutionManager()
        backups = manager.state_manager.list_backups()

        if backups:
            print("💾 BACKUPS:")
            for backup in backups[:5]:  # Show last 5
                created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(backup['created_at']))
                print(f"📁 {backup['path']}")
                print(f"   Transaction: {backup['transaction_id']}")
                print(f"   Created: {created_time}")
                print()
        else:
            print("No backups found")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
