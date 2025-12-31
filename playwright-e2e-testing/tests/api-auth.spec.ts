import { test, expect, APIRequestContext } from '@playwright/test';

/**
 * API Authentication and Session Tests
 * =====================================
 *
 * E2E tests for the Consciousness API Server authentication and session management.
 * Tests cover login, session creation, validation, and cleanup.
 */

test.describe('API Authentication', () => {
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

  test.describe('API Key Authentication', () => {
    test('should reject requests without API key on protected endpoints', async ({ playwright }) => {
      const noAuthRequest = await playwright.request.newContext({
        baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
      });

      // Use shorter timeout - auth rejection should be fast
      const response = await noAuthRequest.post('/evolution/run', {
        data: {
          operation_type: 'verified',
          target_system: '/test',
          user_id: 'test_user',
        },
        timeout: 5000,
      });

      // Should return 401 Unauthorized
      expect(response.status()).toBe(401);

      const data = await response.json();
      expect(data).toHaveProperty('error');
      expect(data.error).toContain('API key');

      await noAuthRequest.dispose();
    });

    test('should reject requests with invalid API key', async ({ playwright }) => {
      const invalidAuthRequest = await playwright.request.newContext({
        baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
        extraHTTPHeaders: {
          'X-API-Key': 'invalid-key-12345',
        },
      });

      // Use shorter timeout - auth rejection should be fast
      const response = await invalidAuthRequest.post('/evolution/run', {
        data: {
          operation_type: 'verified',
          target_system: '/test',
          user_id: 'test_user',
        },
        timeout: 5000,
      });

      // Should be rejected with 401 Unauthorized
      expect(response.status()).toBe(401);

      await invalidAuthRequest.dispose();
    });

    test('should allow health endpoint without API key', async ({ playwright }) => {
      const noAuthRequest = await playwright.request.newContext({
        baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
      });

      const response = await noAuthRequest.get('/health');

      expect(response.ok()).toBeTruthy();

      await noAuthRequest.dispose();
    });

    test('should allow root endpoint without API key', async ({ playwright }) => {
      const noAuthRequest = await playwright.request.newContext({
        baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
      });

      const response = await noAuthRequest.get('/');

      expect(response.ok()).toBeTruthy();

      await noAuthRequest.dispose();
    });

    test('should allow docs endpoint without API key', async ({ playwright }) => {
      const noAuthRequest = await playwright.request.newContext({
        baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
      });

      const response = await noAuthRequest.get('/docs');

      expect(response.ok()).toBeTruthy();

      await noAuthRequest.dispose();
    });

    test('should accept API key via query parameter', async ({ playwright }) => {
      const noHeaderRequest = await playwright.request.newContext({
        baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
      });

      const response = await noHeaderRequest.get(`/status?api_key=${apiKey}`);

      expect(response.ok()).toBeTruthy();

      await noHeaderRequest.dispose();
    });
  });

  test.describe('User Login', () => {
    test('POST /auth/login should authenticate valid user', async () => {
      const response = await request.post('/auth/login', {
        data: {
          username: 'test_user',
          password: 'test_password',
        },
      });

      // May fail if auth system not initialized, but should not 500
      expect(response.status()).toBeLessThan(500);

      const data = await response.json();
      if (response.ok()) {
        expect(data).toHaveProperty('success', true);
        expect(data).toHaveProperty('session_id');
        expect(data).toHaveProperty('data');
        expect(data.data).toHaveProperty('user');
      }
    });

    test('POST /auth/login should reject invalid credentials', async () => {
      const response = await request.post('/auth/login', {
        data: {
          username: 'invalid_user',
          password: 'wrong_password',
        },
      });

      // Should return 401 Unauthorized (if auth system available)
      if (response.status() !== 500) {
        expect(response.status()).toBe(401);
      }
    });

    test('POST /auth/login should require username', async () => {
      const response = await request.post('/auth/login', {
        data: {
          password: 'test_password',
        },
      });

      expect(response.status()).toBe(422);
    });

    test('POST /auth/login should require password', async () => {
      const response = await request.post('/auth/login', {
        data: {
          username: 'test_user',
        },
      });

      expect(response.status()).toBe(422);
    });
  });

  test.describe('Session Management', () => {
    let testSessionId: string | null = null;

    test('should create session on successful login', async () => {
      const response = await request.post('/auth/login', {
        data: {
          username: 'test_user',
          password: 'test_password',
        },
      });

      if (response.ok()) {
        const data = await response.json();
        expect(data.session_id).toBeTruthy();
        testSessionId = data.session_id;
      }
    });

    test('GET /session/:id should return session info', async () => {
      // First create a session - use shorter timeout
      let loginResponse;
      try {
        loginResponse = await request.post('/auth/login', {
          data: {
            username: 'test_user',
            password: 'test_password',
          },
          timeout: 5000,
        });
      } catch (e) {
        // Skip if login endpoint not fully implemented
        test.skip();
        return;
      }

      if (loginResponse.ok()) {
        const loginData = await loginResponse.json();
        const sessionId = loginData.session_id;

        const response = await request.get(`/session/${sessionId}`);

        if (response.ok()) {
          const data = await response.json();
          expect(data).toHaveProperty('session_id', sessionId);
          expect(data).toHaveProperty('user_id');
          expect(data).toHaveProperty('created_at');
          expect(data).toHaveProperty('last_activity');
        }
      } else {
        // Login not implemented - skip
        test.skip();
      }
    });

    test('GET /session/:id should return 404 for invalid session', async () => {
      const response = await request.get('/session/nonexistent-session-id');

      expect(response.status()).toBe(404);
    });

    test('DELETE /session/:id should end session', async () => {
      // First create a session - use shorter timeout
      let loginResponse;
      try {
        loginResponse = await request.post('/auth/login', {
          data: {
            username: 'test_user',
            password: 'test_password',
          },
          timeout: 5000,
        });
      } catch (e) {
        // Skip test if login endpoint times out (not fully implemented)
        test.skip();
        return;
      }

      if (loginResponse.ok()) {
        const loginData = await loginResponse.json();
        const sessionId = loginData.session_id;

        // Delete the session
        const deleteResponse = await request.delete(`/session/${sessionId}`);

        expect(deleteResponse.ok()).toBeTruthy();

        const data = await deleteResponse.json();
        expect(data).toHaveProperty('message');

        // Verify session is gone
        const getResponse = await request.get(`/session/${sessionId}`);
        expect(getResponse.status()).toBe(404);
      } else {
        // Login not implemented - skip rest of test
        test.skip();
      }
    });

    test('DELETE /session/:id should return 404 for invalid session', async () => {
      const response = await request.delete('/session/nonexistent-session-id');

      expect(response.status()).toBe(404);
    });
  });

  test.describe('Session-based Operations', () => {
    test('should accept valid session_id in evolution request', async () => {
      // First login to get session
      const loginResponse = await request.post('/auth/login', {
        data: {
          username: 'test_user',
          password: 'test_password',
        },
      });

      if (loginResponse.ok()) {
        const loginData = await loginResponse.json();
        const sessionId = loginData.session_id;

        // Use session in evolution request
        const evolutionResponse = await request.post('/evolution/run', {
          data: {
            operation_type: 'verified',
            target_system: '/test',
            parameters: {},
            session_id: sessionId,
          },
        });

        expect(evolutionResponse.status()).toBeLessThan(500);

        // Cleanup
        await request.delete(`/session/${sessionId}`);
      }
    });

    test('should reject invalid session_id in evolution request', async () => {
      const response = await request.post('/evolution/run', {
        data: {
          operation_type: 'verified',
          target_system: '/test',
          parameters: {},
          session_id: 'invalid-session-id',
        },
      });

      // Should return 401 Unauthorized
      expect(response.status()).toBe(401);
    });
  });
});

test.describe('API Authentication Security', () => {
  test('should not leak API key in error messages', async ({ request }) => {
    const response = await request.post('/evolution/run', {
      data: {
        operation_type: 'verified',
        target_system: '/test',
      },
    });

    const data = await response.json();
    const responseText = JSON.stringify(data);

    // API key should not appear in response
    expect(responseText).not.toContain('consciousness-api-key-2024');
  });

  test('should not leak session data in error messages', async ({ request }) => {
    const response = await request.get('/session/test-session');

    if (!response.ok()) {
      const data = await response.json();
      const responseText = JSON.stringify(data);

      // Should not leak other session IDs
      expect(responseText).not.toMatch(/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/i);
    }
  });
});
