try:
    from tqdm.autonotebook import tqdm as _real_tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


def tqdm(*args, **kwargs):
    """Wrapper for tqdm that handles optional dependency.
    
    If tqdm is available, calls the real tqdm function.
    If not available, returns a dummy iterator with a warning (unless disabled).
    """
    if _TQDM_AVAILABLE:
        return _real_tqdm(*args, **kwargs)
    else:
        if kwargs.get('disable', False):
            return args[0] if args else []
        
        import warnings
        warnings.warn(
            "tqdm is not installed. Progress bars are disabled. "
            "Install with: pip install sqg[progress]",
            UserWarning,
            stacklevel=2
        )
        return args[0] if args else []
