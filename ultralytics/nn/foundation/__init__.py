"""Opt-in, training-only Foundation Teacher interfaces."""

from .cache import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_TARGET_SHARD_BYTES,
    FeatureCacheReader,
    FeatureCacheWriter,
    build_cache_key,
    compare_feature_caches,
    verify_feature_cache,
)
from .projectors import P4AlignmentProjector
from .protocol import FoundationFeatures, FoundationTeacher
from .losses import cosine_kd_loss, foreground_token_weights, hybrid_kd_loss, relational_kd_loss
from .taps import StudentFeatureTap
from .teachers import DEFAULT_DINOV3_MODEL, DEFAULT_SIGLIP2_MODEL, DINOv3Teacher, MultiFoundationTeacher, SigLIP2Teacher
from .routing import (
    FoundationTeacherRouter,
    foundation_multiteacher_summary,
    foundation_teacher_summary,
    routing_kd_loss,
)
from .semantic import (
    RegionSemanticProjector,
    positive_region_pool,
    region_image_loss,
    region_text_loss,
    semantic_distillation_loss,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_TARGET_SHARD_BYTES",
    "FeatureCacheReader",
    "FeatureCacheWriter",
    "build_cache_key",
    "compare_feature_caches",
    "verify_feature_cache",
    "DEFAULT_DINOV3_MODEL",
    "DINOv3Teacher",
    "DEFAULT_SIGLIP2_MODEL",
    "SigLIP2Teacher",
    "MultiFoundationTeacher",
    "FoundationFeatures",
    "FoundationTeacher",
    "P4AlignmentProjector",
    "StudentFeatureTap",
    "cosine_kd_loss",
    "foreground_token_weights",
    "hybrid_kd_loss",
    "relational_kd_loss",
    "FoundationTeacherRouter",
    "foundation_multiteacher_summary",
    "foundation_teacher_summary",
    "routing_kd_loss",
    "RegionSemanticProjector",
    "positive_region_pool",
    "region_text_loss",
    "region_image_loss",
    "semantic_distillation_loss",
]
