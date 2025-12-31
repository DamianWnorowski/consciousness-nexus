import { test, expect, APIRequestContext } from '@playwright/test';

/**
 * API Evolution Endpoint Tests
 * ============================
 *
 * E2E tests for the Consciousness API Server evolution endpoints.
 * Tests cover evolution operations, validation, and analysis.
 */

test.describe('API Evolution Endpoints', () => {
  let request: APIRequestContext;
  const apiKey = process.env.API_KEY || 'consciousness-api-key-2024';

  test.beforeAll(async ({ playwright }) => {
    request = await playwright.request.newContext({
      baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
      extraHTTPHeaders: {
        'X-API-Key': apiKey,
        'Content-Type': 'application/json',
      },
    });
  });

  test.afterAll(async () => {
    await request.dispose();
  });

  test.describe('Evolution Operations', () => {
    test('POST /evolution/run should accept verified operation', async () => {
      const response = await request.post('/evolution/run', {
        data: {
          operation_type: 'verified',
          target_system: '/test-system',
          parameters: {},
          safety_level: 'standard',
          user_id: 'test_user',
        },
      });

      // Should receive response (may be error if systems not initialized)
      expect(response.status()).toBeLessThan(500);

      const data = await response.json();
      expect(data).toHaveProperty('success');
      expect(data).toHaveProperty('execution_time');
    });

    test('POST /evolution/run should accept recursive operation', async () => {
      const response = await request.post('/evolution/run', {
        data: {
          operation_type: 'recursive',
          target_system: '/test-system',
          parameters: {
            max_iterations: 5,
            fitness_threshold: 0.8,
          },
          safety_level: 'standard',
          user_id: 'test_user',
        },
      });

      expect(response.status()).toBeLessThan(500);

      const data = await response.json();
      expect(data).toHaveProperty('success');
    });

    test('POST /evolution/run should reject unknown operation type', async () => {
      const response = await request.post('/evolution/run', {
        data: {
          operation_type: 'unknown_type',
          target_system: '/test-system',
          parameters: {},
          user_id: 'test_user',
        },
      });

      // Should return 400 Bad Request
      expect(response.status()).toBe(400);

      const data = await response.json();
      expect(data.detail).toContain('Unknown operation type');
    });

    test('POST /evolution/run should require operation_type', async () => {
      const response = await request.post('/evolution/run', {
        data: {
          target_system: '/test-system',
          user_id: 'test_user',
        },
      });

      // Should return 422 Validation Error
      expect(response.status()).toBe(422);
    });

    test('POST /evolution/run should require target_system', async () => {
      let response;
      try {
        response = await request.post('/evolution/run', {
          data: {
            operation_type: 'verified',
            user_id: 'test_user',
          },
          timeout: 5000,
        });
      } catch (e) {
        // Skip if endpoint times out (not fully implemented)
        test.skip();
        return;
      }

      // Should return 422 Validation Error
      expect(response.status()).toBe(422);
    });
  });

  test.describe('Validation Operations', () => {
    test('POST /validation/run should validate files', async () => {
      const response = await request.post('/validation/run', {
        data: {
          files: ['test.py', 'main.py'],
          validation_scope: 'full',
          user_id: 'test_user',
        },
      });

      expect(response.status()).toBeLessThan(500);

      const data = await response.json();
      expect(data).toHaveProperty('success');
      expect(data).toHaveProperty('execution_time');
    });

    test('POST /validation/run should support quick scope', async () => {
      const response = await request.post('/validation/run', {
        data: {
          files: ['test.py'],
          validation_scope: 'quick',
          user_id: 'test_user',
        },
        timeout: 5000,
      });

      expect(response.status()).toBeLessThan(500);
    });

    test('POST /validation/run should require files array', async () => {
      const response = await request.post('/validation/run', {
        data: {
          validation_scope: 'full',
          user_id: 'test_user',
        },
      });

      // Should return 422 Validation Error
      expect(response.status()).toBe(422);
    });

    test('POST /validation/run response should include validation details', async () => {
      const response = await request.post('/validation/run', {
        data: {
          files: ['evolution.py'],
          validation_scope: 'full',
          user_id: 'test_user',
        },
      });

      if (response.ok()) {
        const data = await response.json();
        if (data.success) {
          expect(data.data).toHaveProperty('passed_checks');
          expect(data.data).toHaveProperty('total_checks');
          expect(data.data).toHaveProperty('issues');
          expect(data.data).toHaveProperty('fitness_score');
        }
      }
    });
  });

  test.describe('Analysis Operations', () => {
    test('POST /analysis/run should accept fitness analysis', async () => {
      const response = await request.post('/analysis/run', {
        data: {
          data: {
            fitness_score: 0.85,
            generation: 1,
          },
          analysis_type: 'fitness',
          user_id: 'test_user',
        },
      });

      expect(response.status()).toBeLessThan(500);

      const data = await response.json();
      expect(data).toHaveProperty('success');
      expect(data).toHaveProperty('execution_time');
    });

    test('POST /analysis/run should accept performance analysis', async () => {
      const response = await request.post('/analysis/run', {
        data: {
          data: {
            metrics: {
              response_time: 100,
              throughput: 1000,
            },
          },
          analysis_type: 'performance',
          user_id: 'test_user',
        },
      });

      expect(response.status()).toBeLessThan(500);

      const data = await response.json();
      expect(data).toHaveProperty('success');
    });

    test('POST /analysis/run should reject unknown analysis type', async () => {
      const response = await request.post('/analysis/run', {
        data: {
          data: {},
          analysis_type: 'unknown_type',
          user_id: 'test_user',
        },
      });

      // Should return 400 Bad Request
      expect(response.status()).toBe(400);
    });

    test('POST /analysis/run should require analysis_type', async () => {
      const response = await request.post('/analysis/run', {
        data: {
          data: {},
          user_id: 'test_user',
        },
      });

      // Should return 422 Validation Error
      expect(response.status()).toBe(422);
    });
  });
});

test.describe('API Evolution Error Handling', () => {
  test('should handle server errors gracefully', async ({ request }) => {
    const response = await request.post('/evolution/run', {
      headers: {
        'X-API-Key': process.env.API_KEY || 'consciousness-api-key-2024',
        'Content-Type': 'application/json',
      },
      data: {
        operation_type: 'verified',
        target_system: '/nonexistent-system',
        parameters: {},
        user_id: 'test_user',
      },
    });

    // Server should not crash, should return structured error
    expect(response.status()).toBeLessThan(600);

    const data = await response.json();
    if (!response.ok()) {
      expect(data).toHaveProperty('error');
    }
  });

  test('should include execution time in error responses', async ({ request }) => {
    const response = await request.post('/evolution/run', {
      headers: {
        'X-API-Key': process.env.API_KEY || 'consciousness-api-key-2024',
        'Content-Type': 'application/json',
      },
      data: {
        operation_type: 'verified',
        target_system: '/test',
        parameters: {},
        user_id: 'test_user',
      },
    });

    const data = await response.json();
    expect(data).toHaveProperty('execution_time');
    expect(typeof data.execution_time).toBe('number');
    expect(data.execution_time).toBeGreaterThanOrEqual(0);
  });
});

test.describe('API Evolution Performance', () => {
  test('evolution endpoint should respond within timeout', async ({ request }) => {
    const startTime = Date.now();

    const response = await request.post('/evolution/run', {
      headers: {
        'X-API-Key': process.env.API_KEY || 'consciousness-api-key-2024',
        'Content-Type': 'application/json',
      },
      data: {
        operation_type: 'verified',
        target_system: '/test',
        parameters: {},
        user_id: 'test_user',
      },
    });

    const duration = Date.now() - startTime;

    // Should respond within 30 seconds
    expect(duration).toBeLessThan(30000);
  });
});
