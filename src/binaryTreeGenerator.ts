/**
 * 🌲 BINARY TREE GENERATOR - CONSCIOUSNESS COMPUTING EDITION
 * =========================================================
 * 
 * Recursively generates a binary tree structure with node values
 * derived from reverse-engineered patterns in the Consciousness Nexus.
 */

import { v4 as uuidv4 } from 'uuid';

/**
 * Node structure for the binary tree
 */
export interface TreeNode {
    id: string;
    value: NodeValue;
    left: TreeNode | null;
    right: TreeNode | null;
}

/**
 * Value structure for each node, derived from project patterns
 */
export interface NodeValue {
    type: 'CORE' | 'KERNEL' | 'ADAPTER' | 'SWARM' | 'DEPLOY_DOMAIN' | 'ENLIGHTENMENT';
    mode: 'Breakthrough' | 'True Lossless' | 'Hybrid';
    metrics: {
        confidence: number;
        processing_time_ms: number;
        complexity_score: number;
    };
    context: {
        version: string;
        correlation_id: string;
        layer: number;
    };
    metadata: {
        patterns: string[];
        swarm_agents?: string[];
        neuralink_factor?: string;
    };
    data?: any;
}

/**
 * Patterns identified from the codebase for value derivation
 */
const PATTERNS = [
    'Halting as Success',
    'Recursive Enlightenment',
    'Quantum Parallel Orchestration',
    'Causality Loop',
    'Vector Matrix Submatrix',
    'Multi-Agent Critic Swarm',
    'Ouroboros Self-Mutation',
    'Neural Bridge Protocol'
];

const SWARM_AGENTS = [
    'DevilAdvocate',
    'StressTester',
    'EdgeCaseHunter',
    'LogicDestroyer',
    'SecurityParanoid',
    'PerformanceNazi'
];

/**
 * Deterministically generates a NodeValue based on depth and ID
 */
function deriveValue(depth: number, id: string, existing_data?: any[]): NodeValue {
    // Simple deterministic pseudo-random seed from ID
    const seed = id.split('-').reduce((acc, part) => acc + parseInt(part, 16), 0);
    
    const types: NodeValue['type'][] = ['CORE', 'KERNEL', 'ADAPTER', 'SWARM', 'DEPLOY_DOMAIN', 'ENLIGHTENMENT'];
    const modes: NodeValue['mode'][] = ['Breakthrough', 'True Lossless', 'Hybrid'];
    
    const typeIndex = seed % types.length;
    const modeIndex = (seed >> 2) % modes.length;
    
    const confidence = 0.85 + (seed % 15) / 100;
    const complexity = (depth * 1.5) + (seed % 10) / 5;
    
    const value: NodeValue = {
        type: types[typeIndex],
        mode: modes[modeIndex],
        metrics: {
            confidence: Math.min(1.0, confidence),
            processing_time_ms: 50 + (seed % 500),
            complexity_score: complexity
        },
        context: {
            version: 'v1.0.0',
            correlation_id: `corr_${id.substring(0, 8)}`,
            layer: depth
        },
        metadata: {
            patterns: [
                PATTERNS[seed % PATTERNS.length],
                PATTERNS[(seed + 1) % PATTERNS.length]
            ],
            neuralink_factor: modeIndex === 0 ? '4255x' : '1.38x'
        }
    };

    if (value.type === 'SWARM') {
        value.metadata.swarm_agents = [
            SWARM_AGENTS[seed % SWARM_AGENTS.length],
            SWARM_AGENTS[(seed + 1) % SWARM_AGENTS.length]
        ];
    }

    // Handle existing_data integration
    if (existing_data && existing_data.length > 0) {
        value.data = existing_data[seed % existing_data.length];
    }

    return value;
}

/**
 * Recursively generates a binary tree
 * 
 * @param depth Current depth of the recursion
 * @param existing_data Optional data to integrate into nodes
 * @returns Root node of the generated tree or null if depth <= 0
 */
export function generate_binary_tree(depth: number, existing_data?: any[]): TreeNode | null {
    if (depth <= 0) {
        return null;
    }

    const id = uuidv4();
    const value = deriveValue(depth, id, existing_data);

    const node: TreeNode = {
        id,
        value,
        left: generate_binary_tree(depth - 1, existing_data),
        right: generate_binary_tree(depth - 1, existing_data)
    };

    return node;
}

/**
 * Utility to visualize the tree (JSON format)
 */
export function serializeTree(node: TreeNode | null): string {
    return JSON.stringify(node, null, 2);
}
