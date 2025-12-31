import { test, expect } from '@playwright/test';

test.describe('Consciousness Suite - Critical Path', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the main application
    await page.goto('/');
  });

  test('@critical - System loads and displays main interface', async ({ page }) => {
    // Verify the main page loads
    await expect(page).toHaveTitle(/Consciousness Nexus/);

    // Check for main UI elements
    await expect(page.locator('text=Consciousness Computing Suite')).toBeVisible();
    await expect(page.locator('[data-testid="system-status"]')).toBeVisible();
  });

  test('@critical - ABYSSAL template execution works', async ({ page }) => {
    // Test ABYSSAL template input
    const templateInput = page.locator('[data-testid="abyssal-input"]');
    await templateInput.fill('ABYSSAL[CODE]("test_component")');

    // Execute template
    await page.click('[data-testid="execute-abyssal"]');

    // Verify execution started
    await expect(page.locator('[data-testid="execution-status"]')).toHaveText('EXECUTING');

    // Wait for completion (with timeout)
    await page.waitForSelector('[data-testid="execution-complete"]', { timeout: 30000 });

    // Verify successful execution
    await expect(page.locator('[data-testid="execution-result"]')).toContainText('SUCCESS');
  });

  test('@critical - Security systems operational', async ({ page }) => {
    // Navigate to security dashboard
    await page.click('[data-testid="security-tab"]');

    // Verify security components are active
    await expect(page.locator('[data-testid="integrity-verifier"]')).toHaveText('ACTIVE');
    await expect(page.locator('[data-testid="value-alignment"]')).toHaveText('ENFORCED');
    await expect(page.locator('[data-testid="containment-protocol"]')).toHaveText('ENGAGED');
  });

  test('@critical - Ultra-recursive thinking accessible', async ({ page }) => {
    // Access ultra-recursive thinking interface
    await page.click('[data-testid="ultra-recursive-tab"]');

    // Verify enlightenment status
    await expect(page.locator('[data-testid="enlightenment-status"]')).toHaveText('ACHIEVED');

    // Check for 2026 innovations
    await expect(page.locator('[data-testid="innovation-count"]')).toContainText('12');
  });

  test('@critical - System fitness score displays', async ({ page }) => {
    // Check system health metrics
    const fitnessScore = page.locator('[data-testid="fitness-score"]');
    await expect(fitnessScore).toBeVisible();

    // Verify score is a reasonable number
    const scoreText = await fitnessScore.textContent();
    const score = parseFloat(scoreText || '0');
    expect(score).toBeGreaterThan(80);
    expect(score).toBeLessThanOrEqual(100);
  });
});
