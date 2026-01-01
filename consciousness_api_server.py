#!/usr/bin/env python3
"""
🛡️ CONSCIOUSNESS API SERVER
============================

Universal REST API for Consciousness Computing Suite.
Makes all AI safety and evolution tools available to ANY programming language,
AI session, or development environment via HTTP.
"""

import asyncio
import json
import time
import os
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import logging
import uuid

# Import observability modules
from consciousness_suite.observability import (
    setup_observability,
    ObservabilityConfig,
    get_tracer,
    get_meter,
)
from consciousness_suite.observability.middleware import ObservabilityMiddleware
from consciousness_suite.observability.prometheus.middleware import (
    get_metrics_handler,
    setup_prometheus,
)
from consciousness_suite.observability.tracing import traced

# Import all consciousness systems
from consciousness_suite import (
    AutoRecursiveChainAI,
    VerifiedEvolutionEngine,
    get_safety_orchestrator,
    initialize_consciousness_suite,
    EvolutionAuthSystem,
    EvolutionValidator,
    OptimizedEvolutionAnalyzer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConsciousnessAPI")

# Global state
active_sessions = {}
server_start_time = time.time()

class APIConfig:
    """API server configuration"""
    HOST = os.getenv("CONSCIOUSNESS_API_HOST", "0.0.0.0")
    PORT = int(os.getenv("CONSCIOUSNESS_API_PORT", "8000"))
    WORKERS = int(os.getenv("CONSCIOUSNESS_API_WORKERS", "4"))
    DEBUG = os.getenv("CONSCIOUSNESS_API_DEBUG", "false").lower() == "true"

    # Security
    API_KEY = os.getenv("CONSCIOUSNESS_API_KEY", "consciousness-api-key-2024")
    ENABLE_AUTH = os.getenv("CONSCIOUSNESS_API_AUTH", "true").lower() == "true"

    # CORS
    ALLOW_ORIGINS = os.getenv("CONSCIOUSNESS_API_CORS", "*").split(",")

# Pydantic models for API
class EvolutionRequest(BaseModel):
    """Request for evolution operations"""
    operation_type: str = Field(..., description="Type of evolution operation")
    target_system: str = Field(..., description="Target system to evolve")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Evolution parameters")
    safety_level: str = Field(default="standard", description="Safety enforcement level")
    user_id: str = Field(default="api_user", description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

class ValidationRequest(BaseModel):
    """Request for validation operations"""
    files: List[str] = Field(..., description="Files to validate")
    validation_scope: str = Field(default="full", description="Validation scope")
    user_id: str = Field(default="api_user", description="User identifier")

class AnalysisRequest(BaseModel):
    """Request for analysis operations"""
    data: Dict[str, Any] = Field(..., description="Data to analyze")
    analysis_type: str = Field(..., description="Type of analysis")
    user_id: str = Field(default="api_user", description="User identifier")

class AuthRequest(BaseModel):
    """Authentication request"""
    username: str
    password: str

class APIResponse(BaseModel):
    """Standard API response"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    session_id: Optional[str] = None
    execution_time: float = 0.0

# FastAPI app
app = FastAPI(
    title="Consciousness Computing Suite API",
    description="Universal REST API for enterprise-grade AI safety and evolution",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Observability middleware - distributed tracing and metrics
app.add_middleware(
    ObservabilityMiddleware,
    service_name="consciousness-nexus-api",
    exclude_paths=["/metrics", "/health", "/ready", "/live", "/docs", "/redoc", "/openapi.json"],
)

# Authentication middleware
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Authentication middleware"""
    if not APIConfig.ENABLE_AUTH:
        return await call_next(request)

    # Skip auth for docs and health endpoints
    if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health", "/"]:
        return await call_next(request)

    # Check API key
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key or api_key != APIConfig.API_KEY:
        return JSONResponse(
            status_code=401,
            content={"success": False, "error": "Invalid or missing API key"}
        )

    return await call_next(request)

# Session management
def create_session(user_id: str) -> str:
    """Create a new session"""
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "user_id": user_id,
        "created_at": time.time(),
        "last_activity": time.time(),
        "operations": []
    }
    return session_id

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data"""
    session = active_sessions.get(session_id)
    if session:
        session["last_activity"] = time.time()
    return session

def cleanup_sessions():
    """Clean up expired sessions"""
    current_time = time.time()
    expired = []

    for session_id, session in active_sessions.items():
        # Expire sessions after 24 hours of inactivity
        if current_time - session["last_activity"] > 86400:
            expired.append(session_id)

    for session_id in expired:
        del active_sessions[session_id]

    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")

# Initialize consciousness suite on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the consciousness suite on server startup"""
    logger.info("[*] Starting Consciousness API Server...")

    try:
        # Initialize observability stack
        observability_config = ObservabilityConfig(
            service_name="consciousness-nexus-api",
            service_version="2.0.0",
            environment=os.getenv("ENVIRONMENT", "development"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
            enable_prometheus=True,
            enable_tracing=True,
            trace_sample_rate=1.0,
            langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        )
        setup_observability(observability_config)
        logger.info("[OK] Observability stack initialized")

        # Initialize Prometheus collectors
        setup_prometheus(observability_config)
        logger.info("[OK] Prometheus collectors initialized")

        # Initialize consciousness suite
        await initialize_consciousness_suite()
        logger.info("[OK] Consciousness Suite initialized successfully")

        # Start background cleanup task
        asyncio.create_task(session_cleanup_task())

    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize Consciousness Suite: {e}")
        raise

async def session_cleanup_task():
    """Background task to clean up expired sessions"""
    while True:
        await asyncio.sleep(3600)  # Clean up every hour
        cleanup_sessions()

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Consciousness Computing Suite API",
        "version": "2.0.0",
        "status": "operational",
        "uptime": time.time() - server_start_time,
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions": len(active_sessions),
        "uptime": time.time() - server_start_time
    }

@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    return {"status": "ready"}

@app.get("/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return get_metrics_handler()()

@app.post("/auth/login", response_model=APIResponse)
async def login(request: AuthRequest):
    """Authenticate user and create session"""
    start_time = time.time()

    try:
        # Get auth system
        orchestrator = await get_safety_orchestrator()
        auth_system = orchestrator.systems.get('auth')

        if not auth_system:
            raise HTTPException(status_code=500, detail="Authentication system not available")

        # Authenticate user
        user = auth_system.authenticate(request.username, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Create session
        session_id = create_session(request.username)

        execution_time = time.time() - start_time
        return APIResponse(
            success=True,
            data={"user": request.username, "roles": list(user.roles)},
            session_id=session_id,
            execution_time=execution_time
        )

    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Login error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=execution_time
        )

@app.post("/evolution/run", response_model=APIResponse)
async def run_evolution(request: EvolutionRequest, background_tasks: BackgroundTasks):
    """Run evolution operation"""
    start_time = time.time()

    try:
        # Validate session if provided
        if request.session_id:
            session = get_session(request.session_id)
            if not session:
                raise HTTPException(status_code=401, detail="Invalid session")
            request.user_id = session["user_id"]

        # Get safety orchestrator
        orchestrator = await get_safety_orchestrator(request.safety_level)

        # Choose evolution engine based on operation type
        if request.operation_type == "verified":
            engine = VerifiedEvolutionEngine()
            operation = engine.evolve_with_verification
        elif request.operation_type == "recursive":
            engine = AutoRecursiveChainAI(**request.parameters)
            operation = engine.run_orchestration
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation type: {request.operation_type}")

        # Execute with safety
        result = await orchestrator.execute_safe_operation(
            f"run_{request.operation_type}_evolution",
            operation,
            context=orchestrator.SafetyContext(
                user_id=request.user_id,
                operation_type=request.operation_type,
                risk_level="high",
                requires_confirmation=False
            )
        )

        execution_time = time.time() - start_time

        if result["success"]:
            return APIResponse(
                success=True,
                data=result["execution_result"],
                session_id=request.session_id,
                execution_time=execution_time
            )
        else:
            return APIResponse(
                success=False,
                error=result.get("error", "Evolution failed"),
                execution_time=execution_time
            )

    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Evolution error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=execution_time
        )

@app.post("/validation/run", response_model=APIResponse)
async def run_validation(request: ValidationRequest):
    """Run validation on files"""
    start_time = time.time()

    try:
        # Get validation system
        orchestrator = await get_safety_orchestrator()
        validator = orchestrator.systems.get('validation')

        if not validator:
            raise HTTPException(status_code=500, detail="Validation system not available")

        # Run validation
        result = await validator.validate_evolution(
            request.files,
            validation_scope=request.validation_scope
        )

        execution_time = time.time() - start_time

        return APIResponse(
            success=result.is_valid,
            data={
                "passed_checks": result.total_checks - len(result.issues),
                "total_checks": result.total_checks,
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "category": issue.category.value,
                        "title": issue.title,
                        "description": issue.description,
                        "file": issue.file_path
                    }
                    for issue in result.issues
                ],
                "warnings": len(result.warnings),
                "fitness_score": result.fitness_score
            },
            execution_time=execution_time
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Validation error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=execution_time
        )

@app.post("/analysis/run", response_model=APIResponse)
async def run_analysis(request: AnalysisRequest):
    """Run analysis operations"""
    start_time = time.time()

    try:
        # Get analysis system
        orchestrator = await get_safety_orchestrator()
        analyzer = orchestrator.systems.get('complexity')

        if not analyzer:
            raise HTTPException(status_code=500, detail="Analysis system not available")

        # Run analysis based on type
        if request.analysis_type == "fitness":
            # Analyze fitness data
            analyzer.add_evolution_cycle_batch([request.data])
            stats = analyzer.get_evolution_statistics_optimized()

            return APIResponse(
                success=True,
                data=stats,
                execution_time=time.time() - start_time
            )

        elif request.analysis_type == "performance":
            # Get performance predictions
            predictions = analyzer.get_performance_predictions()

            return APIResponse(
                success=True,
                data=predictions,
                execution_time=time.time() - start_time
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {request.analysis_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "user_id": session["user_id"],
        "created_at": session["created_at"],
        "last_activity": session["last_activity"],
        "operations_count": len(session["operations"])
    }

@app.delete("/session/{session_id}")
async def logout_session(session_id: str):
    """End session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": "Session ended"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/status")
async def system_status():
    """Get comprehensive system status"""
    try:
        orchestrator = await get_safety_orchestrator()
        status = await orchestrator.get_system_status()

        return {
            "api_server": {
                "status": "operational",
                "uptime": time.time() - server_start_time,
                "active_sessions": len(active_sessions)
            },
            "consciousness_suite": status,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Status error: {e}")
        return {
            "api_server": {"status": "error", "error": str(e)},
            "timestamp": time.time()
        }

# Streaming endpoints for long-running operations
@app.post("/evolution/run/stream")
async def run_evolution_stream(request: EvolutionRequest):
    """Run evolution with streaming updates"""
    async def generate():
        try:
            # Start evolution
            orchestrator = await get_safety_orchestrator(request.safety_level)

            if request.operation_type == "recursive":
                engine = AutoRecursiveChainAI(**request.parameters)

                # Stream progress updates
                for i in range(10):  # Simulate progress updates
                    progress_data = {
                        "stage": f"iteration_{i}",
                        "progress": (i + 1) / 10,
                        "message": f"Processing iteration {i + 1}/10"
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    await asyncio.sleep(0.5)

                # Run actual operation
                result = await orchestrator.execute_safe_operation(
                    "run_recursive_evolution",
                    engine.run_orchestration
                )

                yield f"data: {json.dumps({'complete': True, 'result': result})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

if __name__ == "__main__":
    print("[*] CONSCIOUSNESS COMPUTING SUITE API SERVER")
    print("=" * 50)
    print(f"Host: {APIConfig.HOST}")
    print(f"Port: {APIConfig.PORT}")
    print(f"Workers: {APIConfig.WORKERS}")
    print(f"Debug: {APIConfig.DEBUG}")
    print(f"Auth Enabled: {APIConfig.ENABLE_AUTH}")
    print()
    print("API Endpoints:")
    print("  GET  /           - Server info")
    print("  GET  /health     - Health check")
    print("  GET  /docs       - Interactive API docs")
    print("  GET  /status     - System status")
    print("  POST /auth/login - User authentication")
    print("  POST /evolution/run - Run evolution operations")
    print("  POST /validation/run - Run validation checks")
    print("  POST /analysis/run - Run analysis operations")
    print()
    print("[*] Starting server...")

    uvicorn.run(
        "consciousness_api_server:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        workers=APIConfig.WORKERS,
        reload=APIConfig.DEBUG,
        access_log=True
    )
