// Package consciousness provides Go SDK for Consciousness Computing Suite
package consciousness

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

// Client represents a client for the Consciousness API
type Client struct {
    BaseURL    string
    APIKey     string
    HTTPClient *http.Client
}

// NewClient creates a new Consciousness API client
func NewClient(baseURL, apiKey string) *Client {
    return &Client{
        BaseURL: baseURL,
        APIKey:  apiKey,
        HTTPClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

// EvolutionRequest represents an evolution request
type EvolutionRequest struct {
    OperationType string                 `json:"operation_type"`
    TargetSystem  string                 `json:"target_system"`
    Parameters    map[string]interface{} `json:"parameters,omitempty"`
    SafetyLevel   string                 `json:"safety_level"`
    UserID        string                 `json:"user_id"`
}

// EvolutionResult represents the result of an evolution operation
type EvolutionResult struct {
    EvolutionID   string                 `json:"evolution_id"`
    Status        string                 `json:"status"`
    Results       map[string]interface{} `json:"results"`
    Metrics       EvolutionMetrics       `json:"metrics"`
}

// EvolutionMetrics contains evolution performance metrics
type EvolutionMetrics struct {
    FitnessScore   float64 `json:"fitness_score"`
    ExecutionTime  float64 `json:"execution_time"`
    SafetyChecks   int     `json:"safety_checks"`
    Warnings       []string `json:"warnings"`
}

// RunEvolution runs an evolution operation
func (c *Client) RunEvolution(req EvolutionRequest) (*EvolutionResult, error) {
    jsonData, err := json.Marshal(req)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }

    httpReq, err := http.NewRequest("POST", c.BaseURL+"/evolution/run", bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }

    httpReq.Header.Set("Content-Type", "application/json")
    if c.APIKey != "" {
        httpReq.Header.Set("X-API-Key", c.APIKey)
    }

    resp, err := c.HTTPClient.Do(httpReq)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
    }

    var apiResp struct {
        Success bool             `json:"success"`
        Data    *EvolutionResult `json:"data"`
        Error   string           `json:"error"`
    }

    if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }

    if !apiResp.Success {
        return nil, fmt.Errorf("API error: %s", apiResp.Error)
    }

    return apiResp.Data, nil
}

// Health checks API health
func (c *Client) Health() (map[string]interface{}, error) {
    resp, err := c.HTTPClient.Get(c.BaseURL + "/health")
    if err != nil {
        return nil, fmt.Errorf("health check failed: %w", err)
    }
    defer resp.Body.Close()

    var result map[string]interface{}
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, fmt.Errorf("failed to decode health response: %w", err)
    }

    return result, nil
}
