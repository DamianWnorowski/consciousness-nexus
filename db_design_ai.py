#!/usr/bin/env python3
"""
🗄️ AI-POWERED DATABASE DESIGN SYSTEM
====================================

Optimal database schema design with migrations, indexes, and relationships.
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import argparse

@dataclass
class Entity:
    """Database entity (table)"""
    name: str
    attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class DatabaseSchema:
    """Complete database schema"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    migrations: List[str] = field(default_factory=list)
    seed_data: List[Dict[str, Any]] = field(default_factory=list)

class AIDatabaseDesigner:
    """AI-powered database schema designer"""

    def __init__(self):
        self.schema = DatabaseSchema()

    def analyze_requirements(self, requirements: str) -> Dict[str, Any]:
        """Analyze requirements to extract entities and relationships"""
        print("[*] Analyzing requirements...")

        # Extract entities from requirements text
        entities = self._extract_entities(requirements)
        relationships = self._extract_relationships(requirements, entities)

        return {
            "entities": entities,
            "relationships": relationships,
            "complexity": self._assess_complexity(entities, relationships)
        }

    def _extract_entities(self, requirements: str) -> List[str]:
        """Extract entity names from requirements"""
        # Common patterns for entities
        patterns = [
            r'\b(\w+(?:_\w+)*)s?\b(?=\s+(?:with|have|contain|include))',  # "products with..."
            r'\b(\w+(?:_\w+)*)s?\b(?=\s+(?:and|or|,)\s+\w+\s+and)',  # "users and orders and"
            r'\b(\w+(?:_\w+)*)s?\b(?=\s+(?:management|system|platform))',  # "user management"
        ]

        entities = set()
        requirements_lower = requirements.lower()

        for pattern in patterns:
            matches = re.findall(pattern, requirements_lower)
            entities.update(matches)

        # Filter out common words
        stop_words = {'with', 'have', 'and', 'or', 'the', 'a', 'an', 'for', 'to', 'in', 'on', 'at', 'by', 'system', 'platform', 'management'}
        entities = {entity for entity in entities if entity not in stop_words and len(entity) > 2}

        return list(entities)

    def _extract_relationships(self, requirements: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        req_lower = requirements.lower()

        # Look for relationship patterns
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check for belongs_to/has_many patterns
                if f"{entity1} belong" in req_lower or f"{entity2} belong" in req_lower:
                    if f"{entity1}" in req_lower and f"{entity2}" in req_lower:
                        relationships.append({
                            "from": entity1,
                            "to": entity2,
                            "type": "belongs_to",
                            "cardinality": "many_to_one"
                        })

        return relationships

    def _assess_complexity(self, entities: List[str], relationships: List[Dict[str, Any]]) -> str:
        """Assess complexity of the schema"""
        entity_count = len(entities)
        relationship_count = len(relationships)

        if entity_count <= 3 and relationship_count <= 2:
            return "simple"
        elif entity_count <= 6 and relationship_count <= 5:
            return "moderate"
        else:
            return "complex"

    def generate_schema(self, analysis: Dict[str, Any]) -> DatabaseSchema:
        """Generate complete database schema"""
        print("[+] Generating database schema...")

        # Create entities
        for entity_name in analysis["entities"]:
            entity = self._generate_entity(entity_name, analysis)
            self.schema.entities[entity_name] = entity

        # Add relationships
        for rel in analysis["relationships"]:
            self._add_relationship(rel)

        # Generate migrations
        self._generate_migrations()

        # Generate seed data
        self._generate_seed_data()

        return self.schema

    def _generate_entity(self, name: str, analysis: Dict[str, Any]) -> Entity:
        """Generate a single entity"""
        entity = Entity(name=name)

        # Add common attributes based on entity type
        if name in ['user', 'users']:
            entity.attributes = {
                'id': {'type': 'UUID', 'primary': True, 'default': 'gen_random_uuid()'},
                'email': {'type': 'VARCHAR(255)', 'unique': True, 'nullable': False},
                'password_hash': {'type': 'VARCHAR(255)', 'nullable': False},
                'first_name': {'type': 'VARCHAR(100)', 'nullable': False},
                'last_name': {'type': 'VARCHAR(100)', 'nullable': False},
                'created_at': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'},
                'updated_at': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'}
            }
            entity.indexes = [
                {'columns': ['email'], 'type': 'btree'},
                {'columns': ['created_at'], 'type': 'btree', 'order': 'DESC'}
            ]

        elif name in ['product', 'products']:
            entity.attributes = {
                'id': {'type': 'UUID', 'primary': True, 'default': 'gen_random_uuid()'},
                'name': {'type': 'VARCHAR(255)', 'nullable': False},
                'description': {'type': 'TEXT', 'nullable': True},
                'price': {'type': 'DECIMAL(10,2)', 'nullable': False, 'check': 'price >= 0'},
                'stock_quantity': {'type': 'INTEGER', 'default': 0, 'check': 'stock_quantity >= 0'},
                'category_id': {'type': 'UUID', 'nullable': True, 'references': 'categories(id)'},
                'created_at': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'},
                'updated_at': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'}
            }
            entity.indexes = [
                {'columns': ['name'], 'type': 'gin', 'expression': "to_tsvector('english', name)"},
                {'columns': ['category_id'], 'type': 'btree'},
                {'columns': ['price'], 'type': 'btree'}
            ]

        elif name in ['order', 'orders']:
            entity.attributes = {
                'id': {'type': 'UUID', 'primary': True, 'default': 'gen_random_uuid()'},
                'user_id': {'type': 'UUID', 'nullable': False, 'references': 'users(id)'},
                'status': {'type': 'VARCHAR(50)', 'default': "'pending'", 'check': "status IN ('pending', 'paid', 'shipped', 'delivered', 'cancelled')"},
                'total_amount': {'type': 'DECIMAL(10,2)', 'nullable': False, 'check': 'total_amount >= 0'},
                'shipping_address': {'type': 'JSONB', 'nullable': False},
                'created_at': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'},
                'updated_at': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'}
            }
            entity.indexes = [
                {'columns': ['user_id'], 'type': 'btree'},
                {'columns': ['status'], 'type': 'btree'},
                {'columns': ['created_at'], 'type': 'btree', 'order': 'DESC'}
            ]

        else:
            # Generic entity
            entity.attributes = {
                'id': {'type': 'UUID', 'primary': True, 'default': 'gen_random_uuid()'},
                'name': {'type': 'VARCHAR(255)', 'nullable': False},
                'created_at': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'},
                'updated_at': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'}
            }

        return entity

    def _add_relationship(self, relationship: Dict[str, Any]):
        """Add relationship between entities"""
        from_entity = relationship["from"]
        to_entity = relationship["to"]

        if from_entity in self.schema.entities and to_entity in self.schema.entities:
            # Add foreign key to from_entity
            fk_name = f"{to_entity}_id"
            self.schema.entities[from_entity].attributes[fk_name] = {
                'type': 'UUID',
                'nullable': False,
                'references': f"{to_entity}s(id)"
            }

    def _generate_migrations(self):
        """Generate SQL migration scripts"""
        self.schema.migrations = []

        for entity_name, entity in self.schema.entities.items():
            # Create table
            columns_sql = []
            for attr_name, attr_config in entity.attributes.items():
                col_sql = f"{attr_name} {attr_config['type']}"

                if attr_config.get('primary'):
                    col_sql += " PRIMARY KEY"
                elif not attr_config.get('nullable', True):
                    col_sql += " NOT NULL"

                if 'default' in attr_config:
                    col_sql += f" DEFAULT {attr_config['default']}"

                if 'check' in attr_config:
                    col_sql += f" CHECK ({attr_config['check']})"

                if 'references' in attr_config:
                    col_sql += f" REFERENCES {attr_config['references']} ON DELETE CASCADE"

                columns_sql.append(col_sql)

            create_table = f"""
CREATE TABLE {entity_name}s (
  {','.join(columns_sql)}
);
"""
            self.schema.migrations.append(create_table)

            # Create indexes
            for idx in entity.indexes:
                idx_name = f"idx_{entity_name}_{'_'.join(idx['columns'])}"
                if idx.get('expression'):
                    idx_sql = f"CREATE INDEX {idx_name} ON {entity_name}s USING {idx['type']} ({idx['expression']});"
                else:
                    columns = ', '.join(f"{col} {idx.get('order', '')}".strip() for col in idx['columns'])
                    idx_sql = f"CREATE INDEX {idx_name} ON {entity_name}s ({columns});"
                self.schema.migrations.append(idx_sql)

    def _generate_seed_data(self):
        """Generate sample seed data"""
        self.schema.seed_data = []

        # Generate seed data for each entity
        for entity_name, entity in self.schema.entities.items():
            if entity_name in ['user', 'product', 'order']:
                seed_count = 10 if entity_name == 'user' else 50 if entity_name == 'product' else 20

                for i in range(seed_count):
                    seed_record = self._generate_seed_record(entity_name, i)
                    if seed_record:
                        self.schema.seed_data.append({
                            'table': entity_name,
                            'data': seed_record
                        })

    def _generate_seed_record(self, entity_name: str, index: int) -> Optional[Dict[str, Any]]:
        """Generate a single seed record"""
        if entity_name == 'user':
            return {
                'email': f'user{index}@example.com',
                'password_hash': 'hashed_password_here',
                'first_name': f'First{index}',
                'last_name': f'Last{index}'
            }
        elif entity_name == 'product':
            return {
                'name': f'Product {index}',
                'description': f'Description for product {index}',
                'price': 10.99 + index,
                'stock_quantity': 100 + index * 10
            }
        elif entity_name == 'order':
            return {
                'user_id': f'user-{index % 10 + 1}-id',  # Mock UUID
                'status': 'pending',
                'total_amount': 99.99 + index * 10,
                'shipping_address': '{"street": "123 Main St", "city": "Anytown", "zip": "12345"}'
            }
        return None

    def generate_prisma_schema(self) -> str:
        """Generate Prisma schema"""
        schema_lines = [
            'datasource db {',
            '  provider = "postgresql"',
            '  url      = env("DATABASE_URL")',
            '}',
            '',
            'generator client {',
            '  provider = "prisma-client-js"',
            '}',
            ''
        ]

        # Generate models
        for entity_name, entity in self.schema.entities.items():
            model_lines = [f'model {entity_name.title()} {{']
            model_lines.append('  id        String   @id @default(uuid())')

            for attr_name, attr_config in entity.attributes.items():
                if attr_name == 'id':
                    continue  # Already added

                # Map SQL types to Prisma types
                prisma_type = self._sql_to_prisma_type(attr_config['type'])
                line = f'  {self._camel_case(attr_name)}    {prisma_type}'

                # Add modifiers
                if not attr_config.get('nullable', True):
                    line += '  @unique' if attr_config.get('unique') else ''
                elif attr_config.get('primary'):
                    line = f'  {attr_name}    {prisma_type}  @id @default(uuid())'

                if 'references' in attr_config:
                    # Handle relationships
                    ref_table = attr_config['references'].split('(')[0]
                    if ref_table.endswith('s'):
                        ref_table = ref_table[:-1]  # Remove plural
                    line += f'  @relation(fields: [{self._camel_case(attr_name)}], references: [id])'

                model_lines.append(line)

            # Add relationships
            for rel in entity.relationships:
                if rel['type'] == 'belongs_to':
                    related_entity = rel['to'].title()
                    field_name = rel['to'] + 's'
                    model_lines.append(f'  {field_name}    {related_entity}  @relation(fields: [{rel["to"]}_id], references: [id])')

            model_lines.append('}')
            model_lines.append('')
            schema_lines.extend(model_lines)

        return '\n'.join(schema_lines)

    def _sql_to_prisma_type(self, sql_type: str) -> str:
        """Convert SQL type to Prisma type"""
        type_mapping = {
            'UUID': 'String',
            'VARCHAR(255)': 'String',
            'VARCHAR(100)': 'String',
            'TEXT': 'String',
            'DECIMAL(10,2)': 'Decimal',
            'INTEGER': 'Int',
            'TIMESTAMP': 'DateTime',
            'JSONB': 'Json'
        }
        return type_mapping.get(sql_type, 'String')

    def _camel_case(self, snake_str: str) -> str:
        """Convert snake_case to camelCase"""
        components = snake_str.split('_')
        return components[0] + ''.join(word.title() for word in components[1:])

    def export_schema(self, output_dir: str = "database_schema"):
        """Export complete schema to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export SQL migrations
        with open(f"{output_dir}/migrations.sql", 'w') as f:
            f.write("-- Database Migrations\n")
            f.write("-- Generated by AI Database Designer\n\n")
            for migration in self.schema.migrations:
                f.write(migration)
                f.write("\n")

        # Export Prisma schema
        with open(f"{output_dir}/schema.prisma", 'w') as f:
            f.write(self.generate_prisma_schema())

        # Export seed data
        with open(f"{output_dir}/seed.json", 'w') as f:
            json.dump(self.schema.seed_data, f, indent=2)

        print(f"[+] Schema exported to {output_dir}/")
        print(f"   - migrations.sql: SQL migration scripts")
        print(f"   - schema.prisma: Prisma ORM schema")
        print(f"   - seed.json: Sample seed data")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI-Powered Database Design")
    parser.add_argument("requirements", help="Requirements description")
    parser.add_argument("--output-dir", default="database_schema", help="Output directory")
    parser.add_argument("--export", action="store_true", help="Export schema to files")

    args = parser.parse_args()

    designer = AIDatabaseDesigner()

    # Analyze requirements
    analysis = designer.analyze_requirements(args.requirements)
    print(f"[+] Analysis: {len(analysis['entities'])} entities, {len(analysis['relationships'])} relationships")
    print(f"   Complexity: {analysis['complexity']}")
    print(f"   Entities: {', '.join(analysis['entities'])}")

    # Generate schema
    schema = designer.generate_schema(analysis)

    print("\n[+] Generated Schema:")
    print(f"   Entities: {len(schema.entities)}")
    print(f"   Migrations: {len(schema.migrations)}")
    print(f"   Seed Records: {len(schema.seed_data)}")

    # Show sample migration
    if schema.migrations:
        print("\n[+] Sample Migration:")
        print(schema.migrations[0][:200] + "..." if len(schema.migrations[0]) > 200 else schema.migrations[0])

    # Export if requested
    if args.export:
        designer.export_schema(args.output_dir)

if __name__ == "__main__":
    main()
