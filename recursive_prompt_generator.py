#!/usr/bin/env python3
"""
🔄 RECURSIVE PROMPT GENERATOR
==============================

Generate meta-recursive prompts with ungenerator patterns.
"""

import json
import re
import random
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import argparse

class GenerationMode(Enum):
    GENERATE = "generate"    # Create recursive prompt expansions
    ANALYZE = "analyze"      # Analyze prompt structure
    EXPAND = "expand"        # Expand existing prompt into variations

@dataclass
class RecursivePrompt:
    """A recursive prompt with meta-patterns"""
    content: str
    depth: int = 1
    patterns: List[str] = field(default_factory=list)
    ungenerators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptExpansion:
    """Expanded prompt with recursive elements"""
    original: str
    expansions: List[RecursivePrompt] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)

class RecursivePromptGenerator:
    """Generator for meta-recursive prompts with ungenerator patterns"""

    def __init__(self):
        self.pattern_categories = self._define_pattern_categories()
        self.ungenerator_patterns = self._define_ungenerator_patterns()

    def _define_pattern_categories(self) -> Dict[str, List[str]]:
        """Define known endpoint timeout patterns"""
        return {
            "known_endpoint_timeout": [
                "patterns that lead to predictable termination",
                "recursive structures with guaranteed endpoints",
                "self-limiting expansion mechanisms",
                "bounded recursion with escape conditions"
            ],
            "known_cache_miss": [
                "patterns that invalidate cached assumptions",
                "unexpected input handling mechanisms",
                "cache-busting recursive transformations",
                "dynamic pattern adaptation triggers"
            ],
            "unknown_cascade_recovery": [
                "recovery patterns for unknown cascade failures",
                "adaptive response to unexpected recursion depth",
                "cascade failure mitigation strategies",
                "unknown state recovery mechanisms"
            ],
            "unknown_emergent_optimization": [
                "emergent optimization from chaotic inputs",
                "self-organizing patterns from random data",
                "adaptive optimization without predefined rules",
                "emergent efficiency from recursive chaos"
            ]
        }

    def _define_ungenerator_patterns(self) -> List[str]:
        """Define ungenerator patterns that break recursive loops"""
        return [
            "Introduce deliberate imperfection to prevent infinite loops",
            "Add entropy injection points that randomize recursion paths",
            "Create self-destruct mechanisms that terminate after N iterations",
            "Implement pattern recognition that breaks recursive similarity",
            "Add external input validation that can halt recursion",
            "Create meta-awareness that monitors and controls recursion depth",
            "Introduce time-based termination conditions",
            "Add resource consumption limits that force early termination",
            "Create pattern divergence mechanisms that break recursive similarity",
            "Implement feedback loops that modify recursion parameters dynamically"
        ]

    def generate(self, query: str, depth: int = 3, mode: GenerationMode = GenerationMode.GENERATE) -> PromptExpansion:
        """Generate recursive prompts based on mode"""

        expansion = PromptExpansion(original=query)

        if mode == GenerationMode.GENERATE:
            expansion = self._generate_expansions(query, depth)
        elif mode == GenerationMode.ANALYZE:
            expansion.analysis = self._analyze_prompt(query)
        elif mode == GenerationMode.EXPAND:
            expansion = self._expand_existing(query, depth)

        return expansion

    def _generate_expansions(self, query: str, depth: int) -> PromptExpansion:
        """Generate recursive prompt expansions"""
        expansion = PromptExpansion(original=query)

        # Generate multiple recursive prompts at different depths
        for current_depth in range(1, depth + 1):
            prompt = self._create_recursive_prompt(query, current_depth)
            expansion.expansions.append(prompt)

        # Analyze the generated prompts
        expansion.analysis = self._analyze_generated_set(expansion.expansions)

        return expansion

    def _create_recursive_prompt(self, query: str, depth: int) -> RecursivePrompt:
        """Create a single recursive prompt at given depth"""
        prompt = RecursivePrompt(content=query, depth=depth)

        # Add meta-recursive patterns
        prompt.patterns = self._select_patterns(depth)

        # Add ungenerator mechanisms
        prompt.ungenerators = self._select_ungenerators(depth)

        # Build the recursive prompt content
        content_parts = [f"RECURSIVE PROMPT DEPTH {depth}", "=" * 30, f"Original Query: {query}", ""]

        # Add recursive instructions
        content_parts.extend([
            "META-RECURSIVE INSTRUCTIONS:",
            f"1. Analyze the query at depth {depth}",
            f"2. Apply {len(prompt.patterns)} recursive patterns",
            f"3. Use {len(prompt.ungenerators)} ungenerator mechanisms",
            f"4. Generate recursive expansion with bounded termination",
            ""
        ])

        # Add patterns
        content_parts.append("RECURSIVE PATTERNS:")
        for i, pattern in enumerate(prompt.patterns, 1):
            content_parts.append(f"{i}. {pattern}")
        content_parts.append("")

        # Add ungenerators
        content_parts.append("UNGENERATOR MECHANISMS:")
        for i, ungen in enumerate(prompt.ungenerators, 1):
            content_parts.append(f"{i}. {ungen}")
        content_parts.append("")

        # Add recursive execution
        content_parts.extend([
            "EXECUTION PROTOCOL:",
            "- Start with base analysis",
            "- Apply patterns recursively",
            "- Monitor for termination conditions",
            "- Generate meta-insights",
            "- Prevent infinite recursion through ungenerators",
            ""
        ])

        prompt.content = "\n".join(content_parts)
        prompt.metadata = {
            "patterns_count": len(prompt.patterns),
            "ungenerators_count": len(prompt.ungenerators),
            "estimated_complexity": depth * 2.5,
            "termination_guarantees": True
        }

        return prompt

    def _select_patterns(self, depth: int) -> List[str]:
        """Select appropriate patterns for given depth"""
        selected = []

        # Select from different categories based on depth
        categories = list(self.pattern_categories.keys())

        for category in categories[:depth]:
            patterns = self.pattern_categories[category]
            # Select 1-2 patterns from each category
            num_select = min(2, len(patterns))
            selected.extend(random.sample(patterns, num_select))

        return selected[:depth * 2]  # Limit based on depth

    def _select_ungenerators(self, depth: int) -> List[str]:
        """Select ungenerator mechanisms"""
        num_select = min(depth + 1, len(self.ungenerator_patterns))
        return random.sample(self.ungenerator_patterns, num_select)

    def _analyze_prompt(self, query: str) -> Dict[str, Any]:
        """Analyze prompt structure and characteristics"""
        analysis = {
            "length": len(query),
            "complexity_score": self._calculate_complexity(query),
            "recursive_potential": self._assess_recursive_potential(query),
            "pattern_matches": self._find_pattern_matches(query),
            "suggested_depth": self._suggest_depth(query),
            "ungenerator_needs": self._assess_ungenerator_needs(query)
        }

        return analysis

    def _calculate_complexity(self, query: str) -> float:
        """Calculate complexity score (0-1)"""
        score = 0.0

        # Length factor
        score += min(len(query) / 500, 0.3)

        # Word complexity
        words = query.split()
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        score += min(avg_word_length / 10, 0.2)

        # Technical terms
        tech_terms = ['algorithm', 'recursion', 'pattern', 'meta', 'analysis', 'optimization']
        tech_count = sum(1 for term in tech_terms if term in query.lower())
        score += min(tech_count * 0.1, 0.3)

        # Question complexity
        question_count = query.count('?')
        score += min(question_count * 0.1, 0.2)

        return min(score, 1.0)

    def _assess_recursive_potential(self, query: str) -> float:
        """Assess how well the query lends itself to recursive treatment"""
        recursive_indicators = [
            'how', 'why', 'what if', 'analyze', 'understand', 'explain',
            'recursive', 'meta', 'pattern', 'complex', 'deep'
        ]

        indicator_count = sum(1 for indicator in recursive_indicators
                            if indicator in query.lower())

        return min(indicator_count * 0.15, 1.0)

    def _find_pattern_matches(self, query: str) -> List[str]:
        """Find matching patterns in the query"""
        matches = []

        query_lower = query.lower()
        for category, patterns in self.pattern_categories.items():
            for pattern in patterns:
                # Simple keyword matching
                keywords = pattern.split()
                match_count = sum(1 for keyword in keywords
                                if keyword in query_lower)
                if match_count >= len(keywords) * 0.6:  # 60% match
                    matches.append(f"{category}: {pattern}")

        return matches[:5]  # Limit results

    def _suggest_depth(self, query: str) -> int:
        """Suggest appropriate recursion depth"""
        complexity = self._calculate_complexity(query)
        recursive_potential = self._assess_recursive_potential(query)

        base_depth = 2
        complexity_bonus = int(complexity * 3)
        recursive_bonus = int(recursive_potential * 2)

        return max(1, min(base_depth + complexity_bonus + recursive_bonus, 5))

    def _assess_ungenerator_needs(self, query: str) -> Dict[str, Any]:
        """Assess what ungenerator mechanisms are needed"""
        needs = {
            "termination_risk": "high" if len(query) > 200 else "medium",
            "chaos_potential": "high" if "complex" in query.lower() else "low",
            "recommended_ungenerators": 2,
            "primary_concerns": []
        }

        if len(query) > 300:
            needs["primary_concerns"].append("length-induced infinite loops")
        if "recursive" in query.lower():
            needs["primary_concerns"].append("self-referential recursion")
        if "?" in query:
            needs["primary_concerns"].append("open-ended exploration")

        needs["recommended_ungenerators"] = len(needs["primary_concerns"]) + 1

        return needs

    def _analyze_generated_set(self, prompts: List[RecursivePrompt]) -> Dict[str, Any]:
        """Analyze a set of generated prompts"""
        analysis = {
            "total_prompts": len(prompts),
            "depth_range": f"{min(p.depth for p in prompts)}-{max(p.depth for p in prompts)}",
            "average_patterns": sum(len(p.patterns) for p in prompts) / len(prompts),
            "average_ungenerators": sum(len(p.ungenerators) for p in prompts) / len(prompts),
            "complexity_progression": [p.metadata.get("estimated_complexity", 0) for p in prompts],
            "pattern_coverage": self._calculate_pattern_coverage(prompts),
            "termination_guarantees": all(p.metadata.get("termination_guarantees", False) for p in prompts)
        }

        return analysis

    def _calculate_pattern_coverage(self, prompts: List[RecursivePrompt]) -> float:
        """Calculate what percentage of available patterns are used"""
        all_patterns = set()
        for prompt in prompts:
            all_patterns.update(prompt.patterns)

        total_available = sum(len(patterns) for patterns in self.pattern_categories.values())
        coverage = len(all_patterns) / total_available if total_available > 0 else 0

        return min(coverage, 1.0)

    def _expand_existing(self, query: str, depth: int) -> PromptExpansion:
        """Expand an existing prompt into variations"""
        expansion = PromptExpansion(original=query)

        # Create variations at different depths
        for d in range(1, depth + 1):
            # Variation 1: Add meta-layer
            meta_prompt = RecursivePrompt(
                content=f"META ANALYSIS: {query}\n\nAnalyze this prompt recursively at depth {d}, identifying self-referential elements and recursive potential.",
                depth=d,
                patterns=["meta-analysis patterns", "self-reference detection"],
                ungenerators=["depth limiting", "analysis termination"]
            )

            # Variation 2: Pattern expansion
            pattern_prompt = RecursivePrompt(
                content=f"PATTERN EXPANSION: {query}\n\nExpand this query using recursive patterns, applying {d} levels of pattern transformation.",
                depth=d,
                patterns=self._select_patterns(d),
                ungenerators=["pattern divergence", "expansion limiting"]
            )

            # Variation 3: Ungenerator focus
            ungen_prompt = RecursivePrompt(
                content=f"UNGENERATOR ANALYSIS: {query}\n\nApply ungenerator mechanisms to prevent infinite recursion while analyzing this query at depth {d}.",
                depth=d,
                patterns=["recursion prevention"],
                ungenerators=self._select_ungenerators(d)
            )

            expansion.expansions.extend([meta_prompt, pattern_prompt, ungen_prompt])

        expansion.analysis = self._analyze_prompt(query)

        return expansion

    def show_patterns(self):
        """Show available pattern categories"""
        print("[*] AVAILABLE PATTERN CATEGORIES")
        print("=" * 40)

        for category, patterns in self.pattern_categories.items():
            print(f"\n{category.upper()}:")
            for i, pattern in enumerate(patterns, 1):
                print(f"  {i}. {pattern}")

        print(f"\n🎯 UNGENERATOR PATTERNS ({len(self.ungenerator_patterns)}):")
        for i, ungen in enumerate(self.ungenerator_patterns, 1):
            print(f"  {i}. {ungen}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Recursive Prompt Generator")
    parser.add_argument("query", nargs="?", help="Query to generate prompts for")
    parser.add_argument("--depth", type=int, default=3, help="Recursion depth")
    parser.add_argument("--mode", choices=[m.value for m in GenerationMode],
                       default="generate", help="Generation mode")
    parser.add_argument("--patterns", action="store_true", help="Show available patterns")

    args = parser.parse_args()

    generator = RecursivePromptGenerator()

    if args.patterns:
        generator.show_patterns()
        return

    if not args.query:
        print("[-] No query provided. Use --patterns to see available patterns.")
        return

    mode = GenerationMode(args.mode)

    print(f"[*] RECURSIVE PROMPT GENERATOR - {mode.value.upper()}")
    print(f"Query: {args.query}")
    print(f"Depth: {args.depth}")
    print("=" * 60)

    # Generate prompts
    expansion = generator.generate(args.query, args.depth, mode)

    if mode == GenerationMode.ANALYZE:
        print("[+] ANALYSIS RESULTS:")
        print(json.dumps(expansion.analysis, indent=2))

    elif mode in [GenerationMode.GENERATE, GenerationMode.EXPAND]:
        print(f"[+] GENERATED {len(expansion.expansions)} RECURSIVE PROMPTS")
        print()

        for i, prompt in enumerate(expansion.expansions, 1):
            print(f"PROMPT {i} (Depth {prompt.depth})")
            print("-" * 40)
            print(prompt.content)
            print(f"\n[+] Patterns: {len(prompt.patterns)}")
            print(f"[+] Ungenerators: {len(prompt.ungenerators)}")
            if prompt.metadata:
                print(f"[+] Metadata: {json.dumps(prompt.metadata, indent=2)}")
            print("\n" + "=" * 60 + "\n")

        if expansion.analysis:
            print("[+] SET ANALYSIS:")
            print(json.dumps(expansion.analysis, indent=2))

if __name__ == "__main__":
    main()
