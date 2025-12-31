/**
 * API Mocking Utilities
 * =====================
 */

import { Page, Route } from '@playwright/test';

export class MockAPI {
  constructor(private page: Page) {}

  async mockSuccess(endpoint: string | RegExp, data: unknown): Promise<void> {
    await this.page.route(endpoint, (route: Route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(data),
      });
    });
  }

  async mockError(endpoint: string | RegExp, status: number, error: unknown): Promise<void> {
    await this.page.route(endpoint, (route: Route) => {
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(error),
      });
    });
  }

  async mockDelay(endpoint: string | RegExp, delay: number, data: unknown): Promise<void> {
    await this.page.route(endpoint, async (route: Route) => {
      await new Promise(resolve => setTimeout(resolve, delay));
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(data),
      });
    });
  }

  async mockSequence(
    endpoint: string | RegExp,
    responses: Array<{ status?: number; data: unknown }>
  ): Promise<void> {
    let callCount = 0;
    await this.page.route(endpoint, (route: Route) => {
      const response = responses[callCount % responses.length];
      callCount++;
      route.fulfill({
        status: response.status || 200,
        contentType: 'application/json',
        body: JSON.stringify(response.data),
      });
    });
  }

  async mockNetworkFailure(endpoint: string | RegExp): Promise<void> {
    await this.page.route(endpoint, (route: Route) => {
      route.abort('failed');
    });
  }

  async mockRateLimit(endpoint: string | RegExp, retryAfter = 60): Promise<void> {
    await this.page.route(endpoint, (route: Route) => {
      route.fulfill({
        status: 429,
        headers: { 'Content-Type': 'application/json', 'Retry-After': retryAfter.toString() },
        body: JSON.stringify({ error: 'Too Many Requests', retryAfter }),
      });
    });
  }

  async clearMocks(): Promise<void> {
    await this.page.unrouteAll();
  }

  async mockHealthy(): Promise<void> {
    await this.mockSuccess('**/health', {
      status: 'healthy',
      version: '2.0.0',
      timestamp: new Date().toISOString(),
    });
  }

  async mockEvolutionRun(result: 'success' | 'failure'): Promise<void> {
    if (result === 'success') {
      await this.mockSuccess('**/evolution/run', {
        success: true,
        data: { evolution_id: 'evo_' + Date.now(), status: 'completed', fitness_score: 0.95 },
      });
    } else {
      await this.mockError('**/evolution/run', 500, { success: false, error: 'Evolution failed' });
    }
  }
}

export function createMockAPI(page: Page): MockAPI {
  return new MockAPI(page);
}
