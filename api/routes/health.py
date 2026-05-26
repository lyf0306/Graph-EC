from fastapi import APIRouter, Request

from api.schemas.response import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    rm = request.app.state.resources
    neo4j_ok = False
    milvus_ok = False
    embedding_ok = False

    try:
        storage = rm.graph_engine.chunk_entity_relation_graph
        if hasattr(storage, "driver"):
            async with storage.driver.session() as session:
                await session.run("RETURN 1")
            neo4j_ok = True
    except Exception:
        pass

    try:
        response = await rm.embed_client.embeddings.create(
            model=rm.config.embedding_model_name,
            input=["health_check"],
        )
        if response.data:
            embedding_ok = True
    except Exception:
        pass

    try:
        if hasattr(rm.graph_engine, "entities_vdb"):
            await rm.graph_engine.entities_vdb.search(query="health_check", top_k=1)
        milvus_ok = True
    except Exception:
        pass

    overall = "healthy" if (neo4j_ok and milvus_ok and embedding_ok) else "degraded"
    return HealthResponse(
        status=overall,
        neo4j=neo4j_ok,
        milvus=milvus_ok,
        embedding_api=embedding_ok,
    )


@router.get("/health/live")
async def liveness_check():
    return {"status": "alive"}
