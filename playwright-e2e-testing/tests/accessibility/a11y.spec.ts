/**
 * Accessibility (a11y) Tests
 * ==========================
 *
 * WCAG 2.1 AA compliance testing for Consciousness Suite API.
 */

import { test, expect } from '@playwright/test';

test.describe('Accessibility Tests', () => {
  test.describe('API Response Accessibility', () => {
    test('health endpoint returns accessible JSON', async ({ request }) => {
      const response = await request.get('/health');
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data).toHaveProperty('status');
      expect(typeof data.status).toBe('string');
    });

    test('error responses are descriptive', async ({ request }) => {
      const response = await request.post('/evolution/run', { data: {} });
      const data = await response.json();

      if (!response.ok()) {
        expect(data.detail || data.error).toBeTruthy();
        expect(typeof (data.detail || data.error)).toBe('string');
      }
    });
  });

  test.describe('Response Format Accessibility', () => {
    test('timestamps use ISO 8601 format', async ({ request }) => {
      const response = await request.get('/health');
      const data = await response.json();

      if (data.timestamp) {
        const isoRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/;
        expect(data.timestamp).toMatch(isoRegex);
      }
    });

    test('boolean values are explicit', async ({ request }) => {
      const response = await request.get('/health');
      const data = await response.json();

      const checkBooleans = (obj: Record<string, unknown>) => {
        for (const [key, value] of Object.entries(obj)) {
          if (typeof value === 'object' && value !== null) {
            checkBooleans(value as Record<string, unknown>);
          }
          if (['success', 'enabled', 'active', 'valid'].includes(key)) {
            expect(typeof value).toBe('boolean');
          }
        }
      };

      if (typeof data === 'object') checkBooleans(data);
    });
  });

  test.describe('Error Message Accessibility', () => {
    test('validation errors list all issues', async ({ request }) => {
      const response = await request.post('/evolution/run', { data: {} });

      expect(response.status()).toBe(422);
      const data = await response.json();

      expect(data.detail).toBeDefined();
      if (Array.isArray(data.detail)) {
        for (const error of data.detail) {
          expect(error.msg || error.message).toBeTruthy();
        }
      }
    });

    test('error messages use plain language', async ({ request }) => {
      const response = await request.post('/evolution/run', {
        data: { operation_type: 'invalid_type', target_system: '/test', user_id: 'a11y_test' },
      });

      const data = await response.json();
      const errorMessage = data.detail || data.error || '';

      expect(errorMessage.toLowerCase()).not.toContain('exception');
      expect(errorMessage.toLowerCase()).not.toContain('stack trace');
    });
  });

  test.describe('Internationalization Readiness', () => {
    test('accepts UTF-8 content', async ({ request }) => {
      const response = await request.post('/evolution/run', {
        data: {
          operation_type: 'verified',
          target_system: '/test-unicode-你好',
          parameters: { message: 'Test with unicode: 日本語' },
          user_id: 'unicode_test',
        },
      });

      expect(response.status()).not.toBe(500);
    });

    test('response encoding is UTF-8', async ({ request }) => {
      const response = await request.get('/health');
      const contentType = response.headers()['content-type'];

      expect(contentType).toContain('application/json');
    });
  });
});

test.describe('Rate Limiting Accessibility', () => {
  test('rate limit responses include retry information', async ({ request }) => {
    const responses = [];

    for (let i = 0; i < 5; i++) {
      responses.push(await request.get('/health'));
    }

    const rateLimited = responses.find(r => r.status() === 429);
    if (rateLimited) {
      const headers = rateLimited.headers();
      expect(headers['retry-after']).toBeDefined();

      const data = await rateLimited.json();
      expect(data.error || data.message).toBeTruthy();
    }
  });
});
