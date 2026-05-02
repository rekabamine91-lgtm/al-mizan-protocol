"""
================================================================================
Al-Mizan Protocol: Constitutional Autograd for Digital Sovereignty
================================================================================
A framework for ethical AI optimization using Al-Qist and Digital Zakat.

This package provides:
    - MizanValue: Constitutional variable with historical integrity
    - ZakatOrchestrator: Collective redistribution across layers
    - adaptive_zakat_rate: Dynamic rate adjustment for minority protection

Key principles:
    - Al-Qist: Anti-tyranny constraint (self-regulation)
    - Digital Zakat: Knowledge redistribution from rich to poor neurons
    - Audit Trail: Full transparency for every justice event

Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.0.0 (June 2026)
License: Apache 2.0
================================================================================
"""

import sys
import warnings

# ============================================================================
# Version Management
# ============================================================================
__version__ = "1.0.0"
__author__ = "Amine Rekab"
__license__ = "Apache 2.0"
__copyright__ = f"Copyright (c) 2026 {__author__}"
__description__ = "Constitutional Autograd for Digital Sovereignty"

# ============================================================================
# Python Version Check
# ============================================================================
if sys.version_info < (3, 8):
    warnings.warn(
        f"Al-Mizan Protocol requires Python 3.8 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}",
        RuntimeWarning,
    )
elif sys.version_info >= (3, 12):
    warnings.warn(
        f"Al-Mizan Protocol has not been fully tested with Python {sys.version_info.major}.{sys.version_info.minor}. "
        f"Expected compatibility issues may occur.",
        RuntimeWarning,
    )

# ============================================================================
# Import Simplification (Core API)
# ============================================================================
# MizanValue: The constitutional variable with historical integrity
from .engine import MizanValue, adaptive_zakat_rate

# ZakatOrchestrator: Collective redistribution orchestrator
from .zakat_manager import ZakatOrchestrator

# ============================================================================
# Export Control (what gets imported with "from almizan import *")
# ============================================================================
__all__ = [
    # Core classes
    "MizanValue",
    "ZakatOrchestrator",
    # Utility functions
    "adaptive_zakat_rate",
    # Version and info
    "__version__",
    "__author__",
    "__license__",
    # Helper functions
    "info",
    "get_audit_example",
]

# ============================================================================
# Package Information Functions
# ============================================================================

def info():
    """Display basic protocol information and status.
    
    Examples
    --------
    >>> import almizan
    >>> almizan.info()
    ⚖️ Al-Mizan Protocol v1.0.0
    ============================================================================
    Constitutional Autograd Engine initialized.
    
    Status:
      ✅ Al-Qist active (anti-tyranny)
      ✅ Zakat active (knowledge redistribution)
      ✅ Audit Trail active (transparency)
    
    Quick Start:
      from almizan import MizanValue
      x = MizanValue(data=5.0, tau_tyr=10.0)
      x.backward()
    
    Documentation: https://github.com/rekabamine91-lgtm/al-mizan-protocol
    ============================================================================
    """
    print("=" * 70)
    print(f"⚖️ Al-Mizan Protocol v{__version__}")
    print("=" * 70)
    print("Constitutional Autograd Engine initialized.")
    print()
    print("Status:")
    print("  ✅ Al-Qist active (anti-tyranny constraint)")
    print("  ✅ Digital Zakat active (knowledge redistribution)")
    print("  ✅ Adaptive Zakat active (minority protection)")
    print("  ✅ Audit Trail active (full transparency)")
    print()
    print("Quick Start:")
    print("  from almizan import MizanValue")
    print("  x = MizanValue(data=5.0, tau_tyr=10.0)")
    print("  x.backward()   # Automatically applies Al-Qist & Zakat")
    print()
    print("Documentation: https://github.com/rekabamine91-lgtm/al-mizan-protocol")
    print("=" * 70)


def get_audit_example():
    """Return an example audit trail for demonstration.
    
    Returns
    -------
    dict
        Example audit trail output from a MizanValue instance.
    
    Examples
    --------
    >>> import almizan
    >>> almizan.get_audit_example()
    {'label': 'example', 'tyranny_count': 2, 'integrity_score': 0.87, ...}
    """
    # Create a sample neuron with some history
    sample = MizanValue(data=10.0, label="example", tau_tyr=10.0)
    sample.grad = 25.0
    sample.apply_al_qist()  # First tyranny event
    sample.grad = 30.0
    sample.apply_al_qist()  # Second tyranny event
    sample.grad = 5.0
    sample.apply_al_qist()  # Normal gradient
    
    # Simulate some zakat (if orchestrator were used)
    sample.zakat_given = 0.034
    sample.zakat_received = 0.012
    
    return sample.get_audit_trail()


def test_installation():
    """Run a quick test to verify the package is working correctly.
    
    Returns
    -------
    bool
        True if the installation is working correctly, False otherwise.
    
    Examples
    --------
    >>> import almizan
    >>> almizan.test_installation()
    ✅ Al-Mizan Protocol v1.0.0 installation verified.
    True
    """
    try:
        # Create two variables
        a = MizanValue(2.0, label="a")
        b = MizanValue(3.0, label="b")
        
        # Compute expression: c = a*b + a*a
        c = a * b + a * a
        c.backward()
        
        # Verify correctness
        expected_a_grad = 7.0  # ∂c/∂a = b + 2a = 3 + 4 = 7
        expected_b_grad = 2.0  # ∂c/∂b = a = 2
        
        if abs(a.grad - expected_a_grad) < 0.001 and abs(b.grad - expected_b_grad) < 0.001:
            print(f"✅ Al-Mizan Protocol v{__version__} installation verified.")
            print(f"   Test: a=2, b=3, c=a*b+a*a")
            print(f"   ∂c/∂a = {a.grad:.4f} (expected {expected_a_grad})")
            print(f"   ∂c/∂b = {b.grad:.4f} (expected {expected_b_grad})")
            return True
        else:
            print("❌ Al-Mizan Protocol installation test FAILED.")
            print(f"   Got ∂c/∂a={a.grad}, expected {expected_a_grad}")
            return False
            
    except Exception as e:
        print(f"❌ Al-Mizan Protocol installation test FAILED with error: {e}")
        return False


# ============================================================================
# Automatic Testing on Import (optional, disabled by default)
# ============================================================================
# To enable automatic test on import, set environment variable:
#   export ALMIZAN_AUTO_TEST=1
# Then uncomment the lines below:

# import os
# if os.environ.get("ALMIZAN_AUTO_TEST") == "1":
#     test_installation()


# ============================================================================
# Package Initialization Message (optional, disabled by default)
# ============================================================================
# To show info on import, set environment variable:
#   export ALMIZAN_SHOW_INFO=1
# Then uncomment the lines below:

# import os
# if os.environ.get("ALMIZAN_SHOW_INFO") == "1":
#     info()


# ============================================================================
# Logging Configuration (optional)
# ============================================================================
import logging

# Create a logger for the almizan package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def enable_logging(level=logging.INFO):
    """Enable logging for the almizan package.
    
    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO)
    
    Examples
    --------
    >>> import almizan
    >>> almizan.enable_logging()
    >>> from almizan import MizanValue
    >>> x = MizanValue(5.0)
    >>> x.backward()
    INFO:almizan:Applying Al-Qist to example
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.info(f"Al-Mizan Protocol v{__version__} logging enabled")


# ============================================================================
# Package Metadata
# ============================================================================
__all__ = [
    "MizanValue",
    "ZakatOrchestrator",
    "adaptive_zakat_rate",
    "__version__",
    "__author__",
    "__license__",
    "info",
    "get_audit_example",
    "test_installation",
    "enable_logging",
]

# ============================================================================
# End of __init__.py
# ============================================================================
