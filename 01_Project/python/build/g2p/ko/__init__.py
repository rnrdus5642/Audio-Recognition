"""Korean G2P module.

Re-exports the implementation. The mapping table (jamo_ipa) is intended
to be imported directly when callers need the mapping primitives.
"""

from .g2p import KoreanG2P

__all__ = ["KoreanG2P"]
