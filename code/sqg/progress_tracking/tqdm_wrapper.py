import warnings

try:
    from tqdm.autonotebook import tqdm as _real_tqdm
    _TQDM_AVAILABLE = True
    _WARNING_SHOWN = False
except ImportError:
    _TQDM_AVAILABLE = False
    _WARNING_SHOWN = False


def tqdm(*args, **kwargs):
    """Wrapper for tqdm that handles optional dependency.
    
    If tqdm is available, calls the real tqdm function.
    If not available, returns the original iterable with a warning (shown only once).
    """
    global _WARNING_SHOWN
    
    if _TQDM_AVAILABLE:
        return _real_tqdm(*args, **kwargs)
    else:
        if not _WARNING_SHOWN:
            warnings.warn(
                "tqdm is not installed. Progress bars are disabled. "
                "Install with: pip install sqg[progress]",
                UserWarning,
                stacklevel=2
            )
            _WARNING_SHOWN = True
        
        return args[0] if args else []
