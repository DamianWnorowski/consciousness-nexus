"""
Configuration Management for Consciousness Computing Suite
==========================================================

Provides centralized configuration management with validation,
environment-specific settings, and dynamic reconfiguration capabilities.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ConfigValidation:
    """Configuration validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

@dataclass
class ConfigSection:
    """Configuration section with metadata"""
    name: str
    data: Dict[str, Any]
    version: str = "1.0.0"
    last_modified: Optional[str] = None
    checksum: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

class ConfigManager:
    """
    Centralized configuration management system.

    Features:
    - Multi-format support (JSON, YAML, environment variables)
    - Configuration validation and schema checking
    - Environment-specific configurations
    - Dynamic reconfiguration
    - Configuration inheritance and overrides
    - Version control and rollback
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.configs: Dict[str, ConfigSection] = {}
        self.environment = os.getenv('CONSCIOUSNESS_ENV', 'development')
        self.validators = {}
        self.listeners = []

        # Load base configurations
        self._load_base_configs()

    def _load_base_configs(self):
        """Load base configuration files"""
        config_files = [
            'core.json',
            'analysis.json',
            'api.json',
            'orchestration.json',
            'knowledge.json',
            'planning.json',
            'queue.json',
            'abyssal.json',
            'meta_parser.json'
        ]

        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                self._load_config_file(config_path)

    def _load_config_file(self, config_path: Path):
        """Load a configuration file"""
        try:
            with open(config_path, encoding='utf-8') as f:
                if config_path.suffix == '.json':
                    data = json.load(f)
                elif config_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            # Calculate checksum
            checksum = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

            section_name = config_path.stem
            self.configs[section_name] = ConfigSection(
                name=section_name,
                data=data,
                last_modified=os.path.getmtime(config_path),
                checksum=checksum
            )

        except Exception as e:
            print(f"Failed to load config {config_path}: {e}")

    def get_config(self, section: str, key: Optional[str] = None,
                   environment: Optional[str] = None) -> Any:
        """
        Get configuration value with environment overrides

        Args:
            section: Configuration section name
            key: Specific key within section (optional)
            environment: Environment to use (defaults to current)

        Returns:
            Configuration value or section data
        """

        env = environment or self.environment

        if section not in self.configs:
            raise ValueError(f"Configuration section not found: {section}")

        config_section = self.configs[section]

        # Apply environment overrides
        config_data = self._apply_environment_overrides(config_section.data, env)

        if key is None:
            return config_data
        else:
            return self._get_nested_value(config_data, key.split('.'))

    def _apply_environment_overrides(self, config_data: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Apply environment-specific overrides"""
        if 'environments' not in config_data:
            return config_data

        env_overrides = config_data['environments'].get(environment, {})

        # Deep merge overrides into base config
        result = config_data.copy()
        result.pop('environments', None)  # Remove environments section

        self._deep_merge(result, env_overrides)

        return result

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Deep merge override dictionary into base"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _get_nested_value(self, data: Dict[str, Any], keys: List[str]) -> Any:
        """Get nested dictionary value by key path"""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise KeyError(f"Configuration key not found: {'.'.join(keys)}")
        return current

    def set_config(self, section: str, key: str, value: Any,
                   validate: bool = True) -> bool:
        """
        Set configuration value with validation

        Args:
            section: Configuration section
            key: Configuration key
            value: New value
            validate: Whether to validate the change

        Returns:
            Success status
        """

        if section not in self.configs:
            self.configs[section] = ConfigSection(section, {})

        # Apply the change
        config_data = self.configs[section].data
        self._set_nested_value(config_data, key.split('.'), value)

        # Validate if requested
        if validate:
            validation = self.validate_config(section)
            if not validation.is_valid:
                # Revert the change
                self._set_nested_value(config_data, key.split('.'), None)  # Remove the key
                return False

        # Update metadata
        self.configs[section].last_modified = str(datetime.now())
        self.configs[section].checksum = hashlib.md5(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest()

        # Notify listeners
        self._notify_listeners(section, key, value)

        # Save to file
        self._save_config(section)

        return True

    def _set_nested_value(self, data: Dict[str, Any], keys: List[str], value: Any):
        """Set nested dictionary value by key path"""
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def validate_config(self, section: str) -> ConfigValidation:
        """Validate configuration section"""

        if section not in self.configs:
            return ConfigValidation(False, [f"Section not found: {section}"], [], [])

        config_data = self.configs[section].data

        errors = []
        warnings = []
        suggestions = []

        # Get validator for this section
        validator = self.validators.get(section)
        if validator:
            validation_result = validator.validate(config_data)
            errors.extend(validation_result.errors)
            warnings.extend(validation_result.warnings)
            suggestions.extend(validation_result.suggestions)

        # Basic validation
        if not isinstance(config_data, dict):
            errors.append("Configuration must be a dictionary")

        # Environment validation
        if 'environments' in config_data:
            envs = config_data['environments']
            if not isinstance(envs, dict):
                errors.append("Environments section must be a dictionary")
            else:
                valid_envs = ['development', 'staging', 'production']
                for env in envs.keys():
                    if env not in valid_envs:
                        warnings.append(f"Unknown environment: {env}")

        return ConfigValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _save_config(self, section: str):
        """Save configuration section to file"""
        config_section = self.configs[section]
        config_path = self.config_dir / f"{section}.json"

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_section.data, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Failed to save config {section}: {e}")

    def add_validator(self, section: str, validator):
        """Add configuration validator for a section"""
        self.validators[section] = validator

    def add_listener(self, listener):
        """Add configuration change listener"""
        self.listeners.append(listener)

    def _notify_listeners(self, section: str, key: str, value: Any):
        """Notify listeners of configuration changes"""
        for listener in self.listeners:
            try:
                asyncio.create_task(listener.on_config_change(section, key, value))
            except Exception as e:
                print(f"Failed to notify listener: {e}")

    def create_backup(self, section: str) -> Optional[str]:
        """Create backup of configuration section"""
        if section not in self.configs:
            return None

        config_path = self.config_dir / f"{section}.json"
        backup_path = self.config_dir / "backups" / f"{section}_{int(time.time())}.json"

        backup_path.parent.mkdir(exist_ok=True)

        try:
            import shutil
            shutil.copy2(config_path, backup_path)
            return str(backup_path)
        except Exception as e:
            print(f"Failed to create backup for {section}: {e}")
            return None

    def restore_backup(self, backup_path: str) -> bool:
        """Restore configuration from backup"""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            return False

        section = backup_file.stem.split('_')[0]  # Extract section name
        config_path = self.config_dir / f"{section}.json"

        try:
            import shutil
            shutil.copy2(backup_file, config_path)
            self._load_config_file(config_path)
            return True
        except Exception as e:
            print(f"Failed to restore backup {backup_path}: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configuration sections"""
        summary = {
            'total_sections': len(self.configs),
            'environment': self.environment,
            'sections': {}
        }

        for name, section in self.configs.items():
            summary['sections'][name] = {
                'version': section.version,
                'last_modified': section.last_modified,
                'checksum': section.checksum[:8] + '...',  # Truncated
                'key_count': len(section.data),
                'has_environments': 'environments' in section.data
            }

        return summary

# Import here to avoid circular imports
import asyncio  # noqa: E402
from datetime import datetime  # noqa: E402
