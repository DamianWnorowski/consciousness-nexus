import { test, expect, APIRequestContext } from '@playwright/test';

/**
 * API Health and Root Endpoint Tests
 * ===================================
 *
 * E2E tests for the Consciousness API Server health and root endpoints.
 * Tests cover basic server availability, health checks, and status endpoints.
 */

test.describe('API Health Endpoints', () => {
  let request: APIRequestContext;

  test.beforeAll(async ({ playwright }) => {
    request = await playwright.request.newContext({
      baseURL: process.env.API_BASE_URL || 'http://localhost:8000',
      extraHTTPHeaders: {
        'X-API-Key': process.env.API_KEY || 'consciousness-api-key-2024',
      },
    });
  });

  test.afterAll(async () => {
    await request.dispose();
  });

  test('GET / should return server info', async () => {
    const response = await request.get('/');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty('message');
    expect(data).toHaveProperty('version');
    expect(data).toHaveProperty('status');
    expect(data.status).toBe('operational');
    expect(data).toHaveProperty('documentation', '/docs');
  });

  test('GET /health should return healthy status', async () => {
    const response = await request.get('/health');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty('status', 'healthy');
    expect(data).toHaveProperty('timestamp');
    expect(data).toHaveProperty('active_sessions');
    expect(data).toHaveProperty('uptime');
    expect(typeof data.uptime).toBe('number');
  });

  test('GET /status should return system status', async () => {
    const response = await request.get('/status');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty('api_server');
    expect(data.api_server).toHaveProperty('status');
    expect(data).toHaveProperty('timestamp');
  });

  test('GET /docs should be accessible', async () => {
    const response = await request.get('/docs');
    // Swagger UI should return HTML
    expect(response.ok()).toBeTruthy();
    const contentType = response.headers()['content-type'];
    expect(contentType).toContain('text/html');
  });

  test('GET /openapi.json should return OpenAPI schema', async () => {
    const response = await request.get('/openapi.json');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty('openapi');
    expect(data).toHaveProperty('info');
    expect(data.info).toHaveProperty('title');
    expect(data).toHaveProperty('paths');
  });
});

test.describe('API Server Uptime', () => {
  test('server should have positive uptime', async ({ request }) => {
    const response = await request.get('/health');
    const data = await response.json();

    expect(data.uptime).toBeGreaterThan(0);
  });

  test('consecutive health checks should show increasing uptime', async ({ request }) => {
    const response1 = await request.get('/health');
    const data1 = await response1.json();

    // Wait a bit
    await new Promise(resolve => setTimeout(resolve, 100));

    const response2 = await request.get('/health');
    const data2 = await response2.json();

    expect(data2.uptime).toBeGreaterThanOrEqual(data1.uptime);
  });
});

test.describe('API Response Headers', () => {
  test('should include CORS headers', async ({ request }) => {
    // CORS headers are only sent when Origin header is present
    const response = await request.get('/health', {
      headers: {
        'Origin': 'http://localhost:3000',
      },
    });
    const headers = response.headers();

    // CORS headers should be present
    expect(headers).toHaveProperty('access-control-allow-origin');
  });

  test('should return JSON content type', async ({ request }) => {
    const response = await request.get('/health');
    const contentType = response.headers()['content-type'];

    expect(contentType).toContain('application/json');
  });
});
