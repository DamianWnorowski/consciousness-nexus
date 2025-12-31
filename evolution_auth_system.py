#!/usr/bin/env python3
"""
🔐 EVOLUTION AUTHENTICATION & AUTHORIZATION SYSTEM
====================================================

Role-based access control for evolution commands with comprehensive security.
"""

import hashlib
import hmac
import json
import os
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import secrets
import argparse

class UserRole(Enum):
    """User roles with escalating permissions"""
    GUEST = "guest"           # Read-only access
    DEVELOPER = "developer"   # Basic evolution access
    LEAD = "lead"            # Advanced evolution + rollback
    ADMIN = "admin"          # Full system control
    SUPERUSER = "superuser"  # Unlimited access

class Permission(Enum):
    """Granular permissions for evolution operations"""
    # Basic permissions
    READ_STATUS = "read_status"
    VIEW_LOGS = "view_logs"

    # Evolution permissions
    TRIGGER_EVOLUTION = "trigger_evolution"
    STOP_EVOLUTION = "stop_evolution"
    MODIFY_CONTRACTS = "modify_contracts"

    # Administrative permissions
    MANAGE_USERS = "manage_users"
    SYSTEM_CONFIG = "system_config"
    FORCE_ROLLBACK = "force_rollback"

    # Dangerous permissions
    BYPASS_SAFETY = "bypass_safety"
    MODIFY_SECURITY = "modify_security"

@dataclass
class User:
    """Authenticated user with roles and permissions"""
    username: str
    roles: Set[UserRole] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    api_key_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None

    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role"""
        return role in self.roles

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions

    def has_any_role(self, roles: Set[UserRole]) -> bool:
        """Check if user has any of the specified roles"""
        return bool(self.roles & roles)

    def has_any_permission(self, permissions: Set[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        return bool(self.permissions & permissions)

@dataclass
class AuthToken:
    """JWT-like authentication token"""
    user_id: str
    issued_at: float
    expires_at: float
    permissions: Set[Permission]
    signature: str

class EvolutionAuthSystem:
    """Comprehensive authentication and authorization for evolution system"""

    def __init__(self, config_file: str = "auth_config.json"):
        self.config_file = config_file
        self.users: Dict[str, User] = {}
        self.active_tokens: Dict[str, AuthToken] = {}
        self.role_permissions = self._define_role_permissions()
        self.security_config = self._load_security_config()

        # Load existing users
        self._load_users()

    def _define_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Define permissions for each role"""
        return {
            UserRole.GUEST: {
                Permission.READ_STATUS,
                Permission.VIEW_LOGS,
            },
            UserRole.DEVELOPER: {
                Permission.READ_STATUS,
                Permission.VIEW_LOGS,
                Permission.TRIGGER_EVOLUTION,
                Permission.STOP_EVOLUTION,
            },
            UserRole.LEAD: {
                Permission.READ_STATUS,
                Permission.VIEW_LOGS,
                Permission.TRIGGER_EVOLUTION,
                Permission.STOP_EVOLUTION,
                Permission.MODIFY_CONTRACTS,
                Permission.FORCE_ROLLBACK,
            },
            UserRole.ADMIN: {
                Permission.READ_STATUS,
                Permission.VIEW_LOGS,
                Permission.TRIGGER_EVOLUTION,
                Permission.STOP_EVOLUTION,
                Permission.MODIFY_CONTRACTS,
                Permission.MANAGE_USERS,
                Permission.SYSTEM_CONFIG,
                Permission.FORCE_ROLLBACK,
            },
            UserRole.SUPERUSER: {
                # All permissions
                Permission.READ_STATUS,
                Permission.VIEW_LOGS,
                Permission.TRIGGER_EVOLUTION,
                Permission.STOP_EVOLUTION,
                Permission.MODIFY_CONTRACTS,
                Permission.MANAGE_USERS,
                Permission.SYSTEM_CONFIG,
                Permission.FORCE_ROLLBACK,
                Permission.BYPASS_SAFETY,
                Permission.MODIFY_SECURITY,
            }
        }

    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        return {
            'token_expiry': 3600,  # 1 hour
            'max_failed_attempts': 5,
            'lockout_duration': 900,  # 15 minutes
            'password_min_length': 12,
            'require_special_chars': True,
            'session_timeout': 28800,  # 8 hours
        }

    def _load_users(self):
        """Load users from storage"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    users_data = data.get('users', {})

                    for username, user_data in users_data.items():
                        user = User(
                            username=username,
                            roles=set(UserRole(role) for role in user_data.get('roles', [])),
                            permissions=set(Permission(perm) for perm in user_data.get('permissions', [])),
                            api_key_hash=user_data.get('api_key_hash'),
                            created_at=user_data.get('created_at', time.time()),
                            last_login=user_data.get('last_login'),
                            failed_attempts=user_data.get('failed_attempts', 0),
                            locked_until=user_data.get('locked_until')
                        )
                        self.users[username] = user

            except Exception as e:
                print(f"Warning: Failed to load users: {e}")

    def _save_users(self):
        """Save users to storage"""
        data = {
            'users': {},
            'last_updated': time.time()
        }

        for username, user in self.users.items():
            data['users'][username] = {
                'roles': [role.value for role in user.roles],
                'permissions': [perm.value for perm in user.permissions],
                'api_key_hash': user.api_key_hash,
                'created_at': user.created_at,
                'last_login': user.last_login,
                'failed_attempts': user.failed_attempts,
                'locked_until': user.locked_until
            }

        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_user(self, username: str, password: str, roles: List[UserRole],
                   creator_username: str) -> bool:
        """Create a new user with specified roles"""

        # Check if creator has permission
        creator = self.users.get(creator_username)
        if not creator or not creator.has_permission(Permission.MANAGE_USERS):
            raise PermissionError("Insufficient permissions to create users")

        # Validate password
        if not self._validate_password(password):
            raise ValueError("Password does not meet security requirements")

        # Check if user already exists
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        # Create user
        user = User(
            username=username,
            roles=set(roles)
        )

        # Set password (hash it)
        user.api_key_hash = self._hash_password(password)

        # Add role-based permissions
        for role in roles:
            if role in self.role_permissions:
                user.permissions.update(self.role_permissions[role])

        self.users[username] = user
        self._save_users()

        print(f"[+] User {username} created with roles: {[r.value for r in roles]}")
        return True

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with password"""

        user = self.users.get(username)
        if not user:
            return None

        # Check if account is locked
        if user.locked_until and time.time() < user.locked_until:
            raise PermissionError("Account is temporarily locked due to failed login attempts")

        # Verify password
        if not self._verify_password(password, user.api_key_hash):
            user.failed_attempts += 1

            # Lock account if too many failed attempts
            if user.failed_attempts >= self.security_config['max_failed_attempts']:
                user.locked_until = time.time() + self.security_config['lockout_duration']
                self._save_users()

            raise PermissionError("Invalid credentials")

        # Successful authentication
        user.failed_attempts = 0
        user.last_login = time.time()
        user.locked_until = None
        self._save_users()

        return user

    def authorize(self, user: User, required_permissions: Set[Permission],
                 operation: str) -> bool:
        """Check if user is authorized for operation"""

        # Check if user has any of the required permissions
        if not user.has_any_permission(required_permissions):
            print(f"[-] Authorization failed for {user.username}: missing permissions for {operation}")
            return False

        # Additional checks based on operation
        if operation == "trigger_evolution":
            # Require recent login for dangerous operations
            if user.last_login and (time.time() - user.last_login) > self.security_config['session_timeout']:
                print(f"[-] Session expired for {user.username}")
                return False

        return True

    def create_token(self, user: User) -> str:
        """Create authentication token for user"""
        token_id = secrets.token_hex(32)
        now = time.time()

        token = AuthToken(
            user_id=user.username,
            issued_at=now,
            expires_at=now + self.security_config['token_expiry'],
            permissions=user.permissions,
            signature=self._sign_token(token_id, user.username)
        )

        self.active_tokens[token_id] = token

        return token_id

    def validate_token(self, token_id: str) -> Optional[User]:
        """Validate authentication token"""

        token = self.active_tokens.get(token_id)
        if not token:
            return None

        # Check expiry
        if time.time() > token.expires_at:
            del self.active_tokens[token_id]
            return None

        # Get user
        user = self.users.get(token.user_id)
        if not user:
            del self.active_tokens[token_id]
            return None

        return user

    def revoke_token(self, token_id: str, requester: User) -> bool:
        """Revoke authentication token"""

        token = self.active_tokens.get(token_id)
        if not token:
            return False

        # Check if requester can revoke this token
        if (requester.username != token.user_id and
            not requester.has_permission(Permission.MANAGE_USERS)):
            raise PermissionError("Cannot revoke token for another user")

        del self.active_tokens[token_id]
        return True

    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.security_config['password_min_length']:
            return False

        if self.security_config['require_special_chars']:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(char in password for char in special_chars):
                return False

        return True

    def _hash_password(self, password: str) -> str:
        """Hash password securely"""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000  # High iteration count for security
        )
        return f"{salt}:{hashed.hex()}"

    def _verify_password(self, password: str, hash_string: str) -> bool:
        """Verify password against hash"""
        if not hash_string:
            return False

        try:
            salt, hash_value = hash_string.split(':')
            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt.encode(),
                100000
            )
            return hmac.compare_digest(computed_hash.hex(), hash_value)
        except:
            return False

    def _sign_token(self, token_id: str, user_id: str) -> str:
        """Sign token for integrity"""
        secret = os.getenv('AUTH_SECRET', 'default-secret-change-in-production')
        message = f"{token_id}:{user_id}"
        return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    def get_user_permissions(self, username: str) -> Set[Permission]:
        """Get all permissions for a user"""
        user = self.users.get(username)
        return user.permissions if user else set()

    def list_users(self, requester: User) -> List[Dict[str, Any]]:
        """List all users (admin only)"""
        if not requester.has_permission(Permission.MANAGE_USERS):
            raise PermissionError("Insufficient permissions")

        return [
            {
                'username': user.username,
                'roles': [role.value for role in user.roles],
                'permissions': [perm.value for perm in user.permissions],
                'last_login': user.last_login,
                'failed_attempts': user.failed_attempts
            }
            for user in self.users.values()
        ]

class EvolutionAuthGuard:
    """Guard class for protecting evolution operations"""

    def __init__(self, auth_system: EvolutionAuthSystem):
        self.auth = auth_system

    def require_auth(self, token: str) -> User:
        """Require valid authentication"""
        user = self.auth.validate_token(token)
        if not user:
            raise PermissionError("Invalid or expired authentication token")
        return user

    def require_permission(self, user: User, permissions: Set[Permission],
                          operation: str):
        """Require specific permissions"""
        if not self.auth.authorize(user, permissions, operation):
            raise PermissionError(f"Insufficient permissions for operation: {operation}")

    def check_evolution_access(self, user: User, operation: str):
        """Check access for evolution operations"""

        permission_map = {
            'trigger_evolution': {Permission.TRIGGER_EVOLUTION},
            'stop_evolution': {Permission.STOP_EVOLUTION},
            'modify_contracts': {Permission.MODIFY_CONTRACTS},
            'force_rollback': {Permission.FORCE_ROLLBACK},
            'system_config': {Permission.SYSTEM_CONFIG},
            'bypass_safety': {Permission.BYPASS_SAFETY},
            'read_status': {Permission.READ_STATUS},
            'view_logs': {Permission.VIEW_LOGS}
        }

        required_perms = permission_map.get(operation, set())
        if required_perms:
            self.require_permission(user, required_perms, operation)

def main():
    """CLI interface for authentication system"""
    parser = argparse.ArgumentParser(description="Evolution Authentication System")
    parser.add_argument("--create-user", nargs=3, metavar=('USERNAME', 'PASSWORD', 'ROLES'),
                       help="Create new user (roles comma-separated)")
    parser.add_argument("--list-users", action="store_true", help="List all users")
    parser.add_argument("--auth-user", nargs=2, metavar=('USERNAME', 'PASSWORD'),
                       help="Authenticate user and get token")
    parser.add_argument("--check-perm", nargs=3, metavar=('TOKEN', 'PERMISSION', 'OPERATION'),
                       help="Check if token has permission for operation")

    args = parser.parse_args()

    auth_system = EvolutionAuthSystem()

    if args.create_user:
        username, password, roles_str = args.create_user
        roles = set(UserRole(role.strip()) for role in roles_str.split(','))

        try:
            auth_system.create_user(username, password, list(roles), "admin")
            print(f"[+] User {username} created successfully")
        except Exception as e:
            print(f"[-] Failed to create user: {e}")

    elif args.list_users:
        # For demo, create a dummy admin user
        admin = User("admin", {UserRole.ADMIN}, {Permission.MANAGE_USERS})
        try:
            users = auth_system.list_users(admin)
            for user in users:
                print(f"User: {user['username']}, Roles: {user['roles']}")
        except Exception as e:
            print(f"[-] Failed to list users: {e}")

    elif args.auth_user:
        username, password = args.auth_user
        try:
            user = auth_system.authenticate(username, password)
            if user:
                token = auth_system.create_token(user)
                print(f"[+] Authentication successful. Token: {token}")
            else:
                print("[-] Authentication failed")
        except Exception as e:
            print(f"[-] Authentication error: {e}")

    elif args.check_perm:
        token, permission, operation = args.check_perm
        try:
            user = auth_system.validate_token(token)
            if user:
                perm = Permission(permission)
                guard = EvolutionAuthGuard(auth_system)
                guard.check_evolution_access(user, operation)
                print(f"[+] Permission check passed for {user.username}")
            else:
                print("[-] Invalid token")
        except Exception as e:
            print(f"[-] Permission check failed: {e}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
