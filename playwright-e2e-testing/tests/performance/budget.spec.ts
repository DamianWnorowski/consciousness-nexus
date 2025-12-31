/**
 * Performance Budget Tests
 * ========================
 *
 * Ensures API performance meets defined budgets.
 */

import { test, expect, APIRequestContext } from '@playwright/test';

test.describe('Performance Budget Tests', () => {
  let request: APIRequestContext;

  test.beforeAll(async ({ playwright }) => {
    request = await playwright.request.newContext({
      baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
      extraHTTPHeaders: {
        'X-API-Key': process.env.API_KEY || 'consciousness-api-key-2024',
        'Content-Type': 'application/json',
      },
    });
  });

  test.afterAll(async () => {
    await request.dispose();
  });

  test.describe('Response Time Budgets', () => {
    test('health endpoint responds within 100ms', async () => {
      const start = Date.now();
      const response = await request.get('/health');
      const duration = Date.now() - start;

      expect(response.ok()).toBeTruthy();
      expect(duration).toBeLessThan(100);
    });

    test('status endpoint responds within 200ms', async () => {
      const start = Date.now();
      const response = await request.get('/status');
      const duration = Date.now() - start;

      expect(response.status()).toBeLessThan(500);
      expect(duration).toBeLessThan(200);
    });

    test('evolution endpoint responds within 5000ms', async () => {
      const start = Date.now();
      const response = await request.post('/evolution/run', {
        data: {
          operation_type: 'verified',
          target_system: '/perf-test',
          parameters: {},
          user_id: 'perf_test',
        },
      });
      const duration = Date.now() - start;

      expect(response.status()).toBeLessThan(500);
      expect(duration).toBeLessThan(5000);
    });
  });

  test.describe('Throughput Budgets', () => {
    test('handles 10 concurrent health checks', async () => {
      const start = Date.now();
      const requests = Array(10)
        .fill(null)
        .map(() => request.get('/health'));
      const responses = await Promise.all(requests);
      const duration = Date.now() - start;

      const successCount = responses.filter(r => r.ok()).length;
      expect(successCount).toBe(10);
      expect(duration).toBeLessThan(1000);
    });

    test('handles 5 concurrent evolution requests', async () => {
      const start = Date.now();
      const requests = Array(5)
        .fill(null)
        .map((_, i) =>
          request.post('/evolution/run', {
            data: {
              operation_type: 'verified',
              target_system: `/perf-test-${i}`,
              parameters: {},
              user_id: 'perf_test',
            },
          })
        );
      const responses = await Promise.all(requests);
      const duration = Date.now() - start;

      const completedCount = responses.filter(r => r.status() < 500).length;
      expect(completedCount).toBe(5);
      expect(duration).toBeLessThan(10000);
    });
  });

  test.describe('Payload Size Budgets', () => {
    test('health response under 1KB', async () => {
      const response = await request.get('/health');
      const body = await response.text();
      expect(body.length).toBeLessThan(1024);
    });

    test('evolution response under 10KB', async () => {
      const response = await request.post('/evolution/run', {
        data: {
          operation_type: 'verified',
          target_system: '/perf-test',
          parameters: {},
          user_id: 'perf_test',
        },
      });
      const body = await response.text();
      expect(body.length).toBeLessThan(10 * 1024);
    });
  });

  test.describe('Consistency Budgets', () => {
    test('response times are consistent (low variance)', async () => {
      const times: number[] = [];

      for (let i = 0; i < 10; i++) {
        const start = Date.now();
        await request.get('/health');
        times.push(Date.now() - start);
      }

      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      const variance = times.reduce((sum, t) => sum + Math.pow(t - avg, 2), 0) / times.length;
      const cv = (Math.sqrt(variance) / avg) * 100;

      expect(cv).toBeLessThan(50);
    });

    test('no performance degradation under load', async () => {
      const baselineStart = Date.now();
      await request.get('/health');
      const baseline = Date.now() - baselineStart;

      const loadRequests = Array(20)
        .fill(null)
        .map(() => request.get('/health'));
      await Promise.all(loadRequests);

      const afterLoadStart = Date.now();
      await request.get('/health');
      const afterLoad = Date.now() - afterLoadStart;

      expect(afterLoad).toBeLessThan(baseline * 3);
    });
  });
});

test.describe('Memory and Resource Tests', () => {
  test('repeated requests do not cause memory leak patterns', async ({ request }) => {
    const times: number[] = [];

    for (let i = 0; i < 50; i++) {
      const start = Date.now();
      await request.get('/health');
      times.push(Date.now() - start);
    }

    const firstHalf = times.slice(0, 25);
    const secondHalf = times.slice(25);

    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    expect(secondAvg).toBeLessThan(firstAvg * 1.5);
  });
});
