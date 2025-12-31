/**
 * 🛡️ CONSCIOUSNESS COMPUTING SUITE - JavaScript SDK
 * ==================================================
 *
 * Universal SDK for accessing Consciousness Computing Suite from JavaScript/TypeScript.
 * Makes enterprise-grade AI safety and evolution tools available in:
 * - Node.js applications
 * - React/Vue/Angular web apps
 * - Electron desktop apps
 * - Deno runtime
 * - Browser environments (with CORS proxy)
 */

import axios, { AxiosInstance } from 'axios';
import WebSocket from 'ws';
import EventEmitter from 'eventemitter3';
// uuid available for future use if needed

export interface ConsciousnessConfig {
  baseURL: string;
  apiKey?: string;
  timeout?: number;
  enableWebSocket?: boolean;
  autoReconnect?: boolean;
}

export interface EvolutionRequest {
  operationType: 'verified' | 'recursive';
  targetSystem: string;
  parameters?: Record<string, any>;
  safetyLevel?: 'minimal' | 'standard' | 'strict' | 'paranoid';
  userId?: string;
  sessionId?: string;
}

export interface ValidationRequest {
  files: string[];
  validationScope?: 'basic' | 'full' | 'comprehensive';
  userId?: string;
}

export interface AnalysisRequest {
  data: Record<string, any>;
  analysisType: 'fitness' | 'performance' | 'security';
  userId?: string;
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  sessionId?: string;
  executionTime: number;
}

export interface EvolutionResult {
  evolutionId: string;
  status: 'completed' | 'failed' | 'running';
  results: Record<string, any>;
  metrics: {
    fitnessScore: number;
    executionTime: number;
    safetyChecks: number;
    warnings: string[];
  };
}

export interface ValidationResult {
  isValid: boolean;
  totalChecks: number;
  passedChecks: number;
  issues: Array<{
    severity: 'low' | 'medium' | 'high' | 'critical';
    category: string;
    title: string;
    description: string;
    file: string;
  }>;
  fitnessScore: number;
  warnings: string[];
}

export class ConsciousnessClient extends EventEmitter {
  private httpClient: AxiosInstance;
  private wsClient?: WebSocket;
  private config: ConsciousnessConfig;
  private sessionId?: string;

  constructor(config: ConsciousnessConfig) {
    super();
    this.config = {
      timeout: 30000,
      enableWebSocket: true,
      autoReconnect: true,
      ...config
    };

    // Initialize HTTP client
    this.httpClient = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.apiKey && { 'X-API-Key': this.config.apiKey })
      }
    });

    // Initialize WebSocket if enabled
    if (this.config.enableWebSocket) {
      this.initializeWebSocket();
    }
  }

  private initializeWebSocket(): void {
    const wsUrl = this.config.baseURL.replace(/^http/, 'ws') + '/ws';
    this.wsClient = new WebSocket(wsUrl, {
      headers: this.config.apiKey ? { 'X-API-Key': this.config.apiKey } : undefined
    });

    this.wsClient.on('open', () => {
      this.emit('connected');
    });

    this.wsClient.on('message', (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());
        this.emit('message', message);
      } catch (error) {
        this.emit('error', error);
      }
    });

    this.wsClient.on('error', (error) => {
      this.emit('error', error);
      if (this.config.autoReconnect) {
        setTimeout(() => this.initializeWebSocket(), 5000);
      }
    });

    this.wsClient.on('close', () => {
      this.emit('disconnected');
      if (this.config.autoReconnect) {
        setTimeout(() => this.initializeWebSocket(), 5000);
      }
    });
  }

  /**
   * Authenticate user and establish session
   */
  async login(username: string, password: string): Promise<APIResponse<{ sessionId: string; roles: string[] }>> {
    try {
      const response = await this.httpClient.post('/auth/login', { username, password });
      const data = response.data as APIResponse<{ sessionId: string; roles: string[] }>;

      if (data.success && data.sessionId) {
        this.sessionId = data.sessionId;
        this.emit('authenticated', data.data);
      }

      return data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || error.message,
        executionTime: 0
      };
    }
  }

  /**
   * Run evolution operation
   */
  async runEvolution(request: EvolutionRequest): Promise<APIResponse<EvolutionResult>> {
    try {
      const evolutionRequest = {
        operation_type: request.operationType,
        target_system: request.targetSystem,
        parameters: request.parameters || {},
        safety_level: request.safetyLevel || 'standard',
        user_id: request.userId || 'js_sdk_user',
        session_id: request.sessionId || this.sessionId
      };

      const response = await this.httpClient.post('/evolution/run', evolutionRequest);
      return response.data as APIResponse<EvolutionResult>;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || error.message,
        executionTime: 0
      };
    }
  }

  /**
   * Run evolution with streaming progress updates
   */
  async runEvolutionStream(request: EvolutionRequest): Promise<AsyncIterable<APIResponse<Partial<EvolutionResult>>>> {
    const evolutionRequest = {
      operation_type: request.operationType,
      target_system: request.targetSystem,
      parameters: request.parameters || {},
      safety_level: request.safetyLevel || 'standard',
      user_id: request.userId || 'js_sdk_user',
      session_id: request.sessionId || this.sessionId
    };

    const response = await this.httpClient.post('/evolution/run/stream', evolutionRequest, {
      responseType: 'stream'
    });

    return this.createAsyncIterator(response.data);
  }

  /**
   * Run validation on files
   */
  async runValidation(request: ValidationRequest): Promise<APIResponse<ValidationResult>> {
    try {
      const validationRequest = {
        files: request.files,
        validation_scope: request.validationScope || 'full',
        user_id: request.userId || 'js_sdk_user'
      };

      const response = await this.httpClient.post('/validation/run', validationRequest);
      return response.data as APIResponse<ValidationResult>;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || error.message,
        executionTime: 0
      };
    }
  }

  /**
   * Run analysis operations
   */
  async runAnalysis(request: AnalysisRequest): Promise<APIResponse<Record<string, any>>> {
    try {
      const analysisRequest = {
        data: request.data,
        analysis_type: request.analysisType,
        user_id: request.userId || 'js_sdk_user'
      };

      const response = await this.httpClient.post('/analysis/run', analysisRequest);
      return response.data as APIResponse<Record<string, any>>;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || error.message,
        executionTime: 0
      };
    }
  }

  /**
   * Get system health status
   */
  async getHealth(): Promise<APIResponse<Record<string, any>>> {
    try {
      const response = await this.httpClient.get('/health');
      return {
        success: true,
        data: response.data,
        executionTime: 0
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        executionTime: 0
      };
    }
  }

  /**
   * Get comprehensive system status
   */
  async getSystemStatus(): Promise<APIResponse<Record<string, any>>> {
    try {
      const response = await this.httpClient.get('/status');
      return {
        success: true,
        data: response.data,
        executionTime: 0
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        executionTime: 0
      };
    }
  }

  /**
   * Send custom request to API
   */
  async customRequest(method: 'GET' | 'POST' | 'PUT' | 'DELETE', endpoint: string, data?: any): Promise<APIResponse> {
    try {
      const response = await this.httpClient.request({
        method,
        url: endpoint,
        data
      });
      return response.data as APIResponse;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || error.message,
        executionTime: 0
      };
    }
  }

  /**
   * Logout and end session
   */
  async logout(): Promise<APIResponse> {
    if (!this.sessionId) {
      return { success: false, error: 'No active session', executionTime: 0 };
    }

    try {
      await this.httpClient.delete(`/session/${this.sessionId}`);
      this.sessionId = undefined;
      this.emit('loggedOut');
      return { success: true, executionTime: 0 };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        executionTime: 0
      };
    }
  }

  private createAsyncIterator(stream: any): AsyncIterable<APIResponse<Partial<EvolutionResult>>> {
    return {
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          return new Promise((resolve) => {
            stream.once('data', (chunk: Buffer) => {
              const lines = chunk.toString().split('\n\n');
              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  try {
                    const data = JSON.parse(line.slice(6));
                    resolve({ value: data, done: data.complete || data.error });
                    return;
                  } catch (error) {
                    // Continue to next line
                  }
                }
              }
              resolve({ value: undefined, done: false });
            });

            stream.once('end', () => {
              resolve({ value: undefined, done: true });
            });
          });
        }
      })
    };
  }

  /**
   * Close connections and cleanup
   */
  close(): void {
    if (this.wsClient) {
      this.wsClient.close();
    }
    this.removeAllListeners();
  }
}

/**
 * Quick setup for common configurations
 */
export class ConsciousnessSDK {
  static createClient(config: ConsciousnessConfig): ConsciousnessClient {
    return new ConsciousnessClient(config);
  }

  static createLocalClient(port: number = 8000, apiKey?: string): ConsciousnessClient {
    return new ConsciousnessClient({
      baseURL: `http://localhost:${port}`,
      apiKey
    });
  }

  static createProductionClient(apiKey: string, baseURL: string = 'https://api.consciousness-suite.com'): ConsciousnessClient {
    return new ConsciousnessClient({
      baseURL,
      apiKey
    });
  }
}

// Export default client factory
export default ConsciousnessSDK;
