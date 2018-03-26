Logging
=======
This package use the standard python logging module to provide some insightfull logging information.
This page gives a basic introduction on how to use it.
For more information on loggers, you can take a look at the `official documentation`_.

Lightnet has 8 different logging levels:

- DEBUG
- INFO
- WARN
- DEPRECATED
- TEST
- TRAIN
- ERROR
- CRITICAL

``TEST`` and ``TRAIN`` are special levels that are used by the :any:`lightnet engine <lightnet.engine.Engine.log>` to differentiate when the network is in training or testing mode.

There are 2 ways to set the logging level with this package. |br|
The first is by using the ``LN_LOGLVL`` environment variable.
Setting this environment variable before running a lightnet script will force lightnet to filter out messages up to that specified level. When setting this variable to ``DEBUG``, the logging messages will print more information about the module they come from. |br|
The second method is by using the :func:`lightnet.logger.setConsoleLevel` function. This has the benefit that it is programmable, but you wont get the extra information by setting the level to ``DEBUG``.

.. rubric:: Example
.. code:: python

  import lightnet as ln
  import logging

  # This line will log all messages to a file.
  # Note: that this also saves messages from other packages using the logging module
  logging.basicConfig(filename='file.log', filemode='w')      

  # This line will only log TRAIN and TEST level messages to a file
  # Note: this logfile will only consider messages from 'lightnet.*' loggers
  filehandler = ln.logger.setLogFile('file.log', filemode='w')

  # This line enables all messages from lightnet to be printed on the console
  # Note: messages that were printed before this line (eg. upon loading the package) will not be printed
  ln.logger.setConsoleLevel(logging.NOTSET)

  # Use this function to enable/disable colored terminal output
  ln.logger.setConsoleColor(False)

  # If you want to use the logging module yourself, and have the same styling as the lightnet logger,
  # you can use this snippet
  log = logging.getLogger('lightnet.choose-a-name-here')
  log.debug('this is a debug message')
  log.info('this is an info message')
  log.warn('this is a warning')
  log.deprecated('This is a deprecation warning')
  log.error('This is an error')
  log.critical('This as a critical error')
  log.train('This is a special logging level that prefixes the message with `TRAIN`')
  log.test('This is a special logging level that prefixes the message with `TEST`')


.. rubric:: API
.. automethod:: lightnet.logger.setConsoleLevel
.. automethod:: lightnet.logger.setConsoleColor
.. automethod:: lightnet.logger.setLogFile


.. include:: ../links.rst
.. _official documentation: https://docs.python.org/3/library/logging.html