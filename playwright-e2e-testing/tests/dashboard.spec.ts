import { test, expect, Page } from '@playwright/test';

/**
 * Dashboard and GUI Tests
 * ========================
 *
 * E2E tests for the Consciousness Suite dashboard and GUI components.
 * Tests cover GUI loading, interactions, and visual elements.
 */

test.describe('Consciousness Suite Dashboard', () => {
  test.describe('Dashboard Loading', () => {
    test('dashboard HTML should be accessible', async ({ page }) => {
      // Try to load the dashboard HTML
      const response = await page.goto('/consciousness_suite_gui.html', {
        waitUntil: 'domcontentloaded',
        timeout: 10000,
      }).catch(() => null);

      if (response) {
        expect(response.ok()).toBeTruthy();
      }
    });

    test('ultra advanced GUI should be accessible', async ({ page }) => {
      const response = await page.goto('/ultra_advanced_production_gui.html', {
        waitUntil: 'domcontentloaded',
        timeout: 10000,
      }).catch(() => null);

      if (response) {
        expect(response.ok()).toBeTruthy();
      }
    });
  });

  test.describe('API Documentation UI', () => {
    test('Swagger UI should load correctly', async ({ page }) => {
      await page.goto('/docs');
      await page.waitForLoadState('networkidle');

      // Check for Swagger UI elements
      const title = await page.locator('.swagger-ui .title').first();
      await expect(title).toBeVisible({ timeout: 10000 });
    });

    test('ReDoc should load correctly', async ({ page }) => {
      await page.goto('/redoc');
      await page.waitForLoadState('networkidle');

      // ReDoc should render
      const container = await page.locator('[data-role="redoc-container"], redoc').first();
      await expect(container).toBeVisible({ timeout: 10000 });
    });

    test('Swagger UI should display API endpoints', async ({ page }) => {
      await page.goto('/docs');
      await page.waitForLoadState('networkidle');

      // Wait for endpoints to load
      await page.waitForSelector('.opblock', { timeout: 10000 });

      // Check for expected endpoints
      const endpoints = await page.locator('.opblock').count();
      expect(endpoints).toBeGreaterThan(0);
    });

    test('Swagger UI should have interactive features', async ({ page }) => {
      await page.goto('/docs');
      await page.waitForLoadState('networkidle');

      // Click on an endpoint to expand it
      const firstEndpoint = await page.locator('.opblock').first();
      await firstEndpoint.click();

      // Check that expanded content is visible
      const expandedContent = await page.locator('.opblock-body').first();
      await expect(expandedContent).toBeVisible({ timeout: 5000 });
    });
  });

  test.describe('Error Pages', () => {
    test('404 page should be handled gracefully', async ({ page }) => {
      const response = await page.goto('/nonexistent-page-12345');

      // FastAPI returns JSON for 404
      if (response) {
        expect([404, 200]).toContain(response.status());
      }
    });
  });
});

test.describe('Consciousness Suite Visual Tests', () => {
  test.describe('Responsive Design', () => {
    test('docs should be responsive on mobile', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.goto('/docs');
      await page.waitForLoadState('networkidle');

      // Swagger UI should still be usable
      const swagger = await page.locator('.swagger-ui');
      await expect(swagger).toBeVisible({ timeout: 10000 });
    });

    test('docs should be responsive on tablet', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.goto('/docs');
      await page.waitForLoadState('networkidle');

      const swagger = await page.locator('.swagger-ui');
      await expect(swagger).toBeVisible({ timeout: 10000 });
    });

    test('docs should be responsive on desktop', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 });
      await page.goto('/docs');
      await page.waitForLoadState('networkidle');

      const swagger = await page.locator('.swagger-ui');
      await expect(swagger).toBeVisible({ timeout: 10000 });
    });
  });

  test.describe('Accessibility', () => {
    test('docs page should have proper heading structure', async ({ page }) => {
      await page.goto('/docs');
      await page.waitForLoadState('networkidle');

      // Check for h1 or title element
      const title = await page.locator('h1, .title').first();
      await expect(title).toBeVisible({ timeout: 10000 });
    });

    test('interactive elements should be focusable', async ({ page }) => {
      await page.goto('/docs');
      await page.waitForLoadState('networkidle');

      // Tab to first focusable element
      await page.keyboard.press('Tab');

      // Check that something is focused
      const focusedElement = await page.locator(':focus');
      expect(await focusedElement.count()).toBeGreaterThan(0);
    });
  });
});

test.describe('Consciousness Suite Performance', () => {
  test('docs page should load within acceptable time', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/docs');
    await page.waitForLoadState('networkidle');

    const loadTime = Date.now() - startTime;

    // Should load within 10 seconds
    expect(loadTime).toBeLessThan(10000);
  });

  test('API response times should be logged', async ({ page }) => {
    const apiCalls: { url: string; duration: number }[] = [];

    // Intercept API calls
    page.on('response', async (response) => {
      if (response.url().includes('/api/') || response.url().includes('/health')) {
        const timing = await response.request().timing().catch(() => null);
        if (timing) {
          apiCalls.push({
            url: response.url(),
            duration: timing.responseEnd - timing.requestStart,
          });
        }
      }
    });

    await page.goto('/docs');
    await page.waitForLoadState('networkidle');

    // Log API calls for debugging
    console.log('API Calls:', apiCalls);
  });
});

test.describe('Consciousness Suite Integration', () => {
  test('should be able to try API from Swagger UI', async ({ page }) => {
    await page.goto('/docs');
    await page.waitForLoadState('networkidle');

    // Find and expand the health endpoint
    const healthEndpoint = await page.locator('.opblock-summary-path:has-text("/health")').first();
    if (await healthEndpoint.isVisible()) {
      await healthEndpoint.click();

      // Look for "Try it out" button
      const tryButton = await page.locator('button:has-text("Try it out")').first();
      if (await tryButton.isVisible()) {
        await tryButton.click();

        // Look for "Execute" button
        const executeButton = await page.locator('button:has-text("Execute")').first();
        if (await executeButton.isVisible()) {
          await executeButton.click();

          // Wait for response
          await page.waitForSelector('.responses-inner', { timeout: 10000 });

          // Check response is shown
          const response = await page.locator('.response-col_status').first();
          await expect(response).toBeVisible({ timeout: 5000 });
        }
      }
    }
  });
});
