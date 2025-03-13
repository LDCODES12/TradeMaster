"""
Enhanced market data stream manager for handling real-time market data.
Implements proper asyncio handling for Streamlit compatibility with
production-ready features and SSL workaround for macOS.
"""

import logging
import threading
import asyncio
import queue
import ssl
import sys
import time
from typing import Dict, Any, Callable, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MarketDataStreamManager:
    """Manages real-time market data streaming with proper asyncio handling"""

    def __init__(self, api_key: str, api_secret: str, disable_ssl_verification: bool = False):
        """Initialize the market data stream manager"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.stream = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.running = False
        self.stream_thread = None
        self.loop = None
        self.symbols = []
        self.disable_ssl_verification = disable_ssl_verification

        # Command queue for thread communication
        self.command_queue = queue.Queue()

        # Reconnection settings
        self.reconnect_delay = 1  # Start with 1 second delay
        self.max_reconnect_delay = 60  # Maximum 60 seconds between retries
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10  # Maximum number of consecutive reconnect attempts

        # Handlers for different data types
        self.handlers = {
            'trade': {},
            'quote': {},
            'bar': {}
        }

    def setup(self, trade_handler: Callable = None,
              quote_handler: Callable = None,
              bar_handler: Callable = None,
              symbols: list = None):
        """
        Set up the market data stream with handlers

        Args:
            trade_handler: Callback for trade updates
            quote_handler: Callback for quote updates
            bar_handler: Callback for bar updates
            symbols: List of symbols to track
        """
        if trade_handler:
            self._register_handler('trade', trade_handler, symbols)

        if quote_handler:
            self._register_handler('quote', quote_handler, symbols)

        if bar_handler:
            self._register_handler('bar', bar_handler, symbols)

        self.symbols = symbols or []

    def _register_handler(self, data_type: str, handler: Callable, symbols: List[str]):
        """Register a handler for a specific data type and symbols"""
        if not asyncio.iscoroutinefunction(handler):
            logger.warning(f"Handler for {data_type} is not a coroutine function. Converting it.")

            async def async_wrapper(*args, **kwargs):
                return handler(*args, **kwargs)

            handler = async_wrapper

        for symbol in symbols or ['*']:  # Use * as wildcard for all symbols
            self.handlers[data_type][symbol] = handler

    def start(self):
        """Start the market data stream in a separate thread"""
        if self.running:
            logger.info("Market data stream already running")
            return True

        try:
            # Reset reconnection attempts
            self.reconnect_attempts = 0

            # Start in a separate thread to avoid asyncio conflicts
            self.stream_thread = threading.Thread(target=self._run_stream_thread, daemon=True)
            self.stream_thread.start()
            self.running = True
            logger.info(f"Market data stream started for {len(self.symbols)} symbols")
            return True
        except Exception as e:
            logger.error(f"Failed to start market data stream: {e}")
            self.running = False
            return False

    def stop(self):
        """Stop the market data stream"""
        if not self.running:
            return

        self.running = False

        # Send stop command to the thread
        self.command_queue.put({"command": "stop"})

        # Give the thread a moment to process the command
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)

        if self.executor:
            self.executor.shutdown(wait=False)

        logger.info("Market data stream stopped")

    def _run_stream_thread(self):
        """Run the stream in a separate thread with its own event loop"""
        try:
            # Create a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Configure SSL for macOS if needed
            if self.disable_ssl_verification:
                self._configure_ssl_for_macos()

            # Start the stream
            self.loop.run_until_complete(self._init_and_run_stream())
        except Exception as e:
            logger.error(f"Error in market data stream thread: {e}")
        finally:
            if self.loop and hasattr(self.loop, 'is_running') and self.loop.is_running():
                self.loop.stop()

            # Clean up any remaining resources
            logger.info("Market data stream thread exiting")
            self.running = False

    def _configure_ssl_for_macos(self):
        """Configure SSL to work around certificate issues on macOS"""
        if sys.platform != 'darwin':
            return

        try:
            # Monkey patch the ssl module's default context
            default_context = ssl._create_default_https_context

            def accept_all_context(*args, **kwargs):
                context = default_context(*args, **kwargs)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                return context

            ssl._create_default_https_context = accept_all_context
            logger.warning("SSL certificate verification disabled - NOT RECOMMENDED FOR PRODUCTION")
        except Exception as e:
            logger.error(f"Failed to configure SSL workaround: {e}")

    async def _init_and_run_stream(self):
        """Initialize and run the stream with proper error handling"""
        while self.running:
            try:
                # Process any pending commands
                self._process_commands()

                # Import here to avoid circular imports
                from alpaca.data.live import StockDataStream

                # Initialize the stream
                self.stream = StockDataStream(self.api_key, self.api_secret)
                logger.info("Initializing market data stream connection")

                # Set up handlers for different data types
                await self._setup_subscriptions()

                # Wait for the connection to establish
                await asyncio.sleep(0.5)

                # Reset reconnection attempts on successful connection
                self.reconnect_attempts = 0
                self.reconnect_delay = 1

                # Create a monitor task to process commands during streaming
                monitor_task = asyncio.create_task(self._monitor_commands())

                # Connect and run the stream
                stream_task = asyncio.create_task(self._run_stream())

                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [monitor_task, stream_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel any pending tasks
                for task in pending:
                    task.cancel()

                if not self.running:
                    logger.info("Stream stopped by command")
                    break

            except Exception as e:
                logger.error(f"Stream error: {e}")

                if not self.running:
                    break

                # Handle reconnection with exponential backoff
                self.reconnect_attempts += 1

                if self.reconnect_attempts > self.max_reconnect_attempts:
                    logger.error(f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached. Giving up.")
                    self.running = False
                    break

                logger.info(f"Reconnecting in {self.reconnect_delay} seconds (attempt {self.reconnect_attempts})")
                await asyncio.sleep(self.reconnect_delay)

                # Exponential backoff with jitter
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

            finally:
                # Clean up the stream
                if hasattr(self, 'stream') and self.stream is not None:
                    await self._close_stream()

    async def _run_stream(self):
        """Run the stream and handle messages"""
        try:
            # This will start a connection and block until the connection is closed
            # We're using a different approach than asyncio.run() here to avoid conflicts
            await self.stream._start_ws()

            # Subscribe to data
            await self._subscribe_all()

            # Process incoming messages
            while self.running:
                try:
                    message = await asyncio.wait_for(self.stream._ws.recv(), 5)
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    # Normal timeout during receiving - allows us to check if we should stop
                    pass
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    raise
        except Exception as e:
            logger.error(f"Stream running error: {e}")
            raise

    async def _process_message(self, message):
        """Process an incoming message and dispatch to handlers"""
        try:
            import msgpack
            msgs = msgpack.unpackb(message)

            for msg in msgs:
                msg_type = msg.get('T')
                symbol = msg.get('S')

                if msg_type == 't' and 'trade' in self.handlers:
                    # Trade message
                    await self._dispatch_message('trade', symbol, msg)
                elif msg_type == 'q' and 'quote' in self.handlers:
                    # Quote message
                    await self._dispatch_message('quote', symbol, msg)
                elif msg_type == 'b' and 'bar' in self.handlers:
                    # Bar message
                    await self._dispatch_message('bar', symbol, msg)
                elif msg_type == 'subscription':
                    logger.info(f"Subscription confirmed: {msg}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _dispatch_message(self, msg_type, symbol, msg):
        """Dispatch a message to the appropriate handler"""
        handler = None

        # Try to find a handler for this specific symbol
        if symbol in self.handlers[msg_type]:
            handler = self.handlers[msg_type][symbol]
        # Fall back to wildcard handler
        elif '*' in self.handlers[msg_type]:
            handler = self.handlers[msg_type]['*']

        if handler:
            try:
                await handler(msg)
            except Exception as e:
                logger.error(f"Error in {msg_type} handler for {symbol}: {e}")

    async def _setup_subscriptions(self):
        """Set up all the necessary subscriptions"""
        for data_type, handlers in self.handlers.items():
            if handlers:
                symbols = list(handlers.keys())
                if '*' in symbols:
                    # Wildcard subscription handled differently
                    symbols.remove('*')
                    # Add default symbols if we have a wildcard
                    if not symbols and not self.symbols:
                        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']
                    # Add symbols from the main list
                    symbols.extend([s for s in self.symbols if s not in symbols])

                logger.info(f"Setting up {data_type} subscription for {len(symbols)} symbols")

                # Set up the subscription in the stream
                if data_type == 'trade':
                    self.stream.subscribe_trades(self._dummy_handler, *symbols)
                elif data_type == 'quote':
                    self.stream.subscribe_quotes(self._dummy_handler, *symbols)
                elif data_type == 'bar':
                    self.stream.subscribe_bars(self._dummy_handler, *symbols)

    async def _dummy_handler(self, msg):
        """Dummy handler - we'll use our own dispatch system"""
        pass

    async def _subscribe_all(self):
        """Send subscription request for all registered data types"""
        if not self.stream or not hasattr(self.stream, '_ws'):
            logger.error("Cannot subscribe: Stream not initialized")
            return

        # Build subscription message
        msg = {
            'action': 'subscribe',
            'trades': [],
            'quotes': [],
            'bars': []
        }

        # Populate with symbols
        for data_type, handlers in self.handlers.items():
            symbols = list(handlers.keys())
            if '*' in symbols:
                symbols.remove('*')
                symbols.extend([s for s in self.symbols if s not in symbols])

            if data_type == 'trade':
                msg['trades'] = symbols
            elif data_type == 'quote':
                msg['quotes'] = symbols
            elif data_type == 'bar':
                msg['bars'] = symbols

        # Send subscription message
        if msg['trades'] or msg['quotes'] or msg['bars']:
            import msgpack
            await self.stream._ws.send(msgpack.packb(msg))
            logger.info(f"Sent subscription request: {msg}")

    async def _close_stream(self):
        """Close the stream connection"""
        if hasattr(self, 'stream') and self.stream and hasattr(self.stream, '_ws'):
            try:
                await self.stream._ws.close()
                logger.info("Closed WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")

    async def _monitor_commands(self):
        """Monitor the command queue for commands from the main thread"""
        while self.running:
            try:
                # Non-blocking check for commands
                self._process_commands()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error monitoring commands: {e}")

    def _process_commands(self):
        """Process any pending commands from the queue"""
        try:
            while not self.command_queue.empty():
                cmd = self.command_queue.get(block=False)
                if cmd.get('command') == 'stop':
                    logger.info("Received stop command")
                    self.running = False
                self.command_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing commands: {e}")

    def is_running(self):
        """Check if the stream is currently running"""
        return self.running


def fix_macos_certificates():
    """
    Fix SSL certificate verification issues on macOS.
    Run this function once to install certificates properly.
    """
    if sys.platform != 'darwin':
        print("This fix is only needed on macOS.")
        return

    print("Attempting to fix SSL certificates for Python on macOS...")

    try:
        import subprocess
        import os

        # Check Python version
        python_version = '.'.join(sys.version.split('.')[:2])
        cert_command = f"/Applications/Python {python_version}/Install Certificates.command"

        if not os.path.exists(cert_command):
            # Try alternative locations
            cert_command = "/Applications/Python/Install Certificates.command"

            if not os.path.exists(cert_command):
                print(f"Certificate installation script not found.")
                print("Manual instructions:")
                print("1. Locate 'Install Certificates.command' in your Python installation folder")
                print("2. Run it by double-clicking in Finder or using terminal")
                print("\nAlternatively, you can run your app with SSL verification disabled:")
                print("   manager = MarketDataStreamManager(api_key, api_secret, disable_ssl_verification=True)")
                return

        # Run the certificate installation script
        print(f"Running certificate installer: {cert_command}")
        subprocess.run([cert_command], check=True, shell=True)
        print("Certificates successfully installed!")

    except Exception as e:
        print(f"Error fixing certificates: {e}")
        print("\nYou can still run your app with SSL verification disabled (not recommended for production):")
        print("   manager = MarketDataStreamManager(api_key, api_secret, disable_ssl_verification=True)")