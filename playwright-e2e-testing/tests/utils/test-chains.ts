/**
 * Multi-Chain Parallel Test Architecture
 * ======================================
 */

import { test as base, Page } from '@playwright/test';

export const testChains = {
  health: ['health-check', 'status-check', 'version-check'],
  authentication: ['login-valid', 'login-invalid', 'session-management', 'logout'],
  evolution: ['evolution-verified', 'evolution-recursive', 'evolution-validation'],
  errors: ['invalid-request', 'missing-params', 'unauthorized', 'rate-limiting'],
  performance: ['response-times', 'concurrent-requests', 'payload-sizes'],
};

interface ChainContext {
  chainId: string;
  testIndex: number;
  sharedState: Record<string, unknown>;
}

type ChainFixtures = { chainContext: ChainContext };

export const chainTest = base.extend<ChainFixtures>({
  chainContext: async ({}, use) => {
    await use({ chainId: 'default', testIndex: 0, sharedState: {} });
  },
});

export async function executeChain(
  chainName: keyof typeof testChains,
  page: Page,
  testRunner: (testName: string, page: Page, context: ChainContext) => Promise<void>
): Promise<{
  passed: number;
  failed: number;
  results: Array<{ test: string; status: 'passed' | 'failed'; error?: string }>;
}> {
  const tests = testChains[chainName];
  const context: ChainContext = { chainId: chainName, testIndex: 0, sharedState: {} };
  const results: Array<{ test: string; status: 'passed' | 'failed'; error?: string }> = [];
  let passed = 0,
    failed = 0;

  for (const testName of tests) {
    try {
      await testRunner(testName, page, context);
      results.push({ test: testName, status: 'passed' });
      passed++;
    } catch (error) {
      results.push({
        test: testName,
        status: 'failed',
        error: error instanceof Error ? error.message : String(error),
      });
      failed++;
    }
    context.testIndex++;
  }

  return { passed, failed, results };
}

export async function executeParallelChains(
  chains: Array<keyof typeof testChains>,
  pageFactory: () => Promise<Page>,
  testRunner: (testName: string, page: Page, context: ChainContext) => Promise<void>
): Promise<
  Map<
    string,
    { passed: number; failed: number; results: Array<{ test: string; status: 'passed' | 'failed'; error?: string }> }
  >
> {
  const results = new Map();
  await Promise.all(
    chains.map(async chainName => {
      const page = await pageFactory();
      try {
        const chainResult = await executeChain(chainName, page, testRunner);
        results.set(chainName, chainResult);
      } finally {
        await page.close();
      }
    })
  );
  return results;
}

export class ChainStateManager {
  private state = new Map<string, Record<string, unknown>>();

  set(chainId: string, key: string, value: unknown): void {
    if (!this.state.has(chainId)) this.state.set(chainId, {});
    this.state.get(chainId)![key] = value;
  }

  get<T>(chainId: string, key: string): T | undefined {
    return this.state.get(chainId)?.[key] as T | undefined;
  }

  clear(chainId: string): void {
    this.state.delete(chainId);
  }

  clearAll(): void {
    this.state.clear();
  }
}

export const chainState = new ChainStateManager();
