/**
 * Edge Case Testing Matrix
 * ========================
 *
 * Comprehensive edge case definitions for zero-error coverage.
 */

export const edgeCases = {
  inputs: {
    empty: '',
    whitespace: '   ',
    specialChars: '!@#$%^&*()_+-=[]{}|;:,.<>?',
    unicode: '你好世界',
    emoji: '🚀🌍💡',
    sqlInjection: "'; DROP TABLE users; --",
    xss: '<script>alert("XSS")</script>',
    longString: 'a'.repeat(10000),
    maxInt: 2147483647,
    negativeInt: -2147483648,
    zero: 0,
  },

  network: {
    slow3G: { latency: 2000, downloadThroughput: (400 * 1024) / 8 },
    offline: { offline: true },
  },

  httpStatus: {
    badRequest: 400,
    unauthorized: 401,
    forbidden: 403,
    notFound: 404,
    tooManyRequests: 429,
    internalError: 500,
    serviceUnavailable: 503,
  },

  viewport: {
    mobile_small: { width: 320, height: 568 },
    tablet: { width: 768, height: 1024 },
    desktop: { width: 1920, height: 1080 },
  },

  auth: {
    noToken: null,
    emptyToken: '',
    invalidToken: 'invalid-token-12345',
    expiredToken: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjB9.invalid',
  },
};

export interface EdgeCaseTest {
  name: string;
  category: string;
  caseName: string;
  value: unknown;
  expectedBehavior?: 'error' | 'success' | 'validation';
}

export function generateEdgeCaseTests(
  component: string,
  action: string,
  categories: (keyof typeof edgeCases)[] = ['inputs']
): EdgeCaseTest[] {
  const tests: EdgeCaseTest[] = [];

  for (const category of categories) {
    const cases = edgeCases[category];
    if (!cases) continue;

    for (const [caseName, value] of Object.entries(cases)) {
      tests.push({
        name: `${component} - ${action} - ${category}:${caseName}`,
        category,
        caseName,
        value,
        expectedBehavior: inferExpectedBehavior(category, caseName),
      });
    }
  }

  return tests;
}

function inferExpectedBehavior(
  category: string,
  caseName: string
): 'error' | 'success' | 'validation' {
  if (category === 'inputs') {
    if (['sqlInjection', 'xss', 'empty', 'whitespace'].includes(caseName)) return 'validation';
    return 'validation';
  }
  if (category === 'network' || category === 'httpStatus' || category === 'auth') return 'error';
  return 'success';
}

export function createEdgeCaseMock(
  category: keyof typeof edgeCases,
  caseName: string
): { status?: number; body?: unknown; delay?: number } {
  if (category === 'httpStatus') {
    return {
      status: edgeCases.httpStatus[caseName as keyof typeof edgeCases.httpStatus],
      body: { error: `Mock ${caseName} error` },
    };
  }
  return { status: 200, body: { success: true } };
}
