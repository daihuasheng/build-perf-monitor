"""
Signal handling for the orchestration module.

This module manages signal registration, cleanup, and delegation to active
BuildRunner instances using a global registry pattern.
"""

import logging
import signal
import threading
from typing import TYPE_CHECKING, Any, Dict

from .shared_state import RuntimeState

if TYPE_CHECKING:
    from .build_runner import BuildRunner

logger = logging.getLogger(__name__)

# Global state management for signal handling
# Since signal handlers cannot be bound to class instances directly,
# we maintain a registry of active runners and their signal handlers.
_active_runners: Dict[int, "BuildRunner"] = {}
_active_runners_lock = threading.Lock()


class SignalHandler:
    """
    Manages signal registration and cleanup for BuildRunner instances.
    
    This class provides a clean interface for signal handling while maintaining
    the global registry required by Python's signal handling mechanism.
    """
    
    def __init__(self, state: RuntimeState):
        self.state = state
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
        self._signal_handlers_set = False
        
    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for this BuildRunner instance."""
        try:
            # Store original handlers so we can restore them later
            self._original_sigint_handler = signal.signal(signal.SIGINT, self._global_signal_handler)
            self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._global_signal_handler)
            self._signal_handlers_set = True
            logger.debug("Signal handlers set up for BuildRunner instance")
        except Exception as e:
            logger.warning(f"Failed to set up signal handlers: {e}")

    def cleanup_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if not self._signal_handlers_set:
            return
        
        try:
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            if self._original_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            logger.debug("Signal handlers restored for BuildRunner instance")
        except Exception as e:
            logger.warning(f"Failed to restore signal handlers: {e}")
        finally:
            self._signal_handlers_set = False

    def register_runner(self, runner_id: int, runner: "BuildRunner") -> None:
        """
        Register a BuildRunner instance for signal handling.
        
        Args:
            runner_id: Unique identifier for the runner instance
            runner: The BuildRunner instance to register
        """
        with _active_runners_lock:
            _active_runners[runner_id] = runner
            logger.debug(f"Registered BuildRunner {runner_id} for signal handling")

    def unregister_runner(self, runner_id: int) -> None:
        """
        Unregister a BuildRunner instance from signal handling.
        
        Args:
            runner_id: Unique identifier for the runner instance to remove
        """
        with _active_runners_lock:
            if runner_id in _active_runners:
                del _active_runners[runner_id]
                logger.debug(f"Unregistered BuildRunner {runner_id} from signal handling")

    @staticmethod
    def _global_signal_handler(signum: int, frame: Any) -> None:
        """
        Global signal handler that delegates to active BuildRunner instances.
        
        This is safer than using a single global variable as it can handle
        multiple BuildRunner instances properly.
        
        Args:
            signum: Signal number that was received
            frame: Current stack frame (unused)
        """
        logger.warning(f"Signal {signum} received. Notifying all active BuildRunner instances.")
        with _active_runners_lock:
            for runner_id, runner in _active_runners.items():
                logger.info(f"Requesting shutdown for BuildRunner {runner_id}")
                # Access the shutdown_requested event through the runner's state
                if hasattr(runner, 'state') and hasattr(runner.state, 'shutdown_requested'):
                    runner.state.shutdown_requested.set()
                else:
                    logger.warning(f"BuildRunner {runner_id} does not have expected state structure")
 