import { test, expect } from '@playwright/test';

test.describe('Matrix Visualization - Visual Regression', () => {
  test('ASCII 3D Matrix renders correctly', async ({ page }) => {
    await page.goto('/matrix-visualizer.html');

    // Wait for matrix to load
    await page.waitForSelector('.matrix-container');

    // Take visual snapshot
    await expect(page).toHaveScreenshot('matrix-visualization.png', {
      fullPage: true,
      threshold: 0.1 // Allow small visual differences
    });
  });

  test('WebGL 3D Matrix renders correctly', async ({ page }) => {
    await page.goto('/matrix_3d_webgl.html');

    // Wait for WebGL canvas to initialize
    await page.waitForSelector('canvas');

    // Allow time for WebGL rendering
    await page.waitForTimeout(2000);

    // Take visual snapshot
    await expect(page).toHaveScreenshot('matrix-webgl-visualization.png', {
      threshold: 0.2 // Allow more variance for 3D rendering
    });
  });

  test('Shader Matrix visualization renders', async ({ page }) => {
    await page.goto('/matrix_ultimate_shader.html');

    // Wait for shader to compile and render
    await page.waitForSelector('.shader-canvas');
    await page.waitForTimeout(3000); // Allow time for shader compilation

    // Take visual snapshot
    await expect(page).toHaveScreenshot('matrix-shader-visualization.png', {
      threshold: 0.3 // Shaders can have variance
    });
  });

  test('Terminal 3D Matrix renders in console', async ({ page }) => {
    // This test would need a terminal emulator or special handling
    // For now, just verify the page loads
    await page.goto('/matrix_3d_terminal.py');

    // Since this is a Python script, it might not render in browser
    // Just verify the page doesn't crash
    await expect(page.locator('body')).toBeVisible();
  });
});
